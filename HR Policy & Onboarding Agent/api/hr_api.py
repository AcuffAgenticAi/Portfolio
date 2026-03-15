"""
================================================================================
api/hr_api.py  —  HR Agent REST API
================================================================================

Purpose:
    FastAPI application exposing the HR Policy Agent, Onboarding Agent, and
    Benefits Agent as REST endpoints.  This is the interface that Teams bots,
    Slack apps, web portals, and mobile apps consume.

Endpoints:
    POST /policy/ask               → Answer an HR policy question (RAG)
    GET  /policy/index-stats       → Knowledge base index statistics
    POST /policy/ingest            → Ingest a new HR document into the KB

    POST /onboarding/plans         → Create an onboarding plan for a new hire
    GET  /onboarding/{employee_id} → Get employee onboarding status
    PATCH /onboarding/{employee_id}/tasks/{task_id}/complete → Complete a task
    PATCH /onboarding/{employee_id}/tasks/{task_id}/block    → Block a task
    GET  /onboarding/coordinator/dashboard                   → Coordinator view

    GET  /benefits/{employee_id}/pto    → PTO balance
    GET  /benefits/retirement-match     → 401(k) match calculation
    GET  /benefits/{employee_id}/summary → Full benefits summary

    GET  /healthz  → liveness
    GET  /readyz   → readiness
================================================================================
"""

from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException, Path, Query
from fastapi.responses import JSONResponse
from pydantic import BaseModel, validator

from agents.hr_policy_agent  import HRPolicyAgent
from agents.onboarding_agent import OnboardingAgent
from agents.benefits_agent   import BenefitsAgent
from agents.knowledge_base_agent import KnowledgeBaseAgent

logger = logging.getLogger("platform.hr_api")

app = FastAPI(
    title       = "HR Policy & Onboarding Agent API",
    description = "RAG-powered HR assistant with onboarding management and benefits calculator.",
    version     = "1.0.0",
)

# ── Agent singletons (initialised at startup) ─────────────────────────────────
_kb:            Optional[KnowledgeBaseAgent] = None
_policy_agent:  Optional[HRPolicyAgent]      = None
_onboarding:    Optional[OnboardingAgent]    = None
_benefits:      Optional[BenefitsAgent]      = None


@app.on_event("startup")
async def startup_event():
    global _kb, _policy_agent, _onboarding, _benefits
    _kb           = KnowledgeBaseAgent()
    _policy_agent = HRPolicyAgent(knowledge_base=_kb)
    _onboarding   = OnboardingAgent()
    _benefits     = BenefitsAgent()
    logger.info("HR Agent API ready")


# ── Pydantic schemas ──────────────────────────────────────────────────────────

class PolicyQuestionRequest(BaseModel):
    question:        str
    employee_id:     Optional[str] = "anonymous"
    conversation_id: Optional[str] = None
    doc_type_filter: Optional[str] = None

    @validator("question")
    def question_not_empty(cls, v):
        if not v.strip():
            raise ValueError("question must not be empty")
        return v.strip()


class IngestRequest(BaseModel):
    filepath: str
    doc_type: str = "policy"


class CreatePlanRequest(BaseModel):
    employee_id:   str
    employee_name: str
    role:          str
    department:    str
    start_date:    str
    level:         str   = "individual_contributor"
    manager_id:    str   = ""

    @validator("start_date")
    def valid_date(cls, v):
        try:
            date.fromisoformat(v)
        except ValueError:
            raise ValueError("start_date must be YYYY-MM-DD format")
        return v


class CompleteTaskRequest(BaseModel):
    notes: str = ""


class BlockTaskRequest(BaseModel):
    reason: str

    @validator("reason")
    def reason_not_empty(cls, v):
        if not v.strip():
            raise ValueError("reason must not be empty")
        return v.strip()


# ── Health endpoints ──────────────────────────────────────────────────────────

@app.get("/healthz", tags=["Health"])
def liveness():
    return {"status": "alive"}


@app.get("/readyz", tags=["Health"])
def readiness():
    if _policy_agent is None:
        raise HTTPException(status_code=503, detail="Agents not initialised")
    return {
        "status":       "ready",
        "kb_chunks":    _kb.index_stats()["total_chunks"] if _kb else 0,
        "onboarding_plans": len(_onboarding._plans) if _onboarding else 0,
    }


# ── Policy / RAG endpoints ────────────────────────────────────────────────────

@app.post("/policy/ask", tags=["HR Policy"])
def ask_policy(body: PolicyQuestionRequest):
    """
    Answer an HR policy question using RAG over the knowledge base.

    Returns the grounded answer with citations, intent classification,
    and a follow-up prompt list.  Sensitive questions are automatically
    escalated to the HR team.
    """
    if _policy_agent is None:
        raise HTTPException(status_code=503, detail="Policy agent not ready")

    response = _policy_agent.answer(
        question        = body.question,
        employee_id     = body.employee_id or "anonymous",
        conversation_id = body.conversation_id,
    )
    return JSONResponse(content=response.to_dict())


@app.get("/policy/index-stats", tags=["HR Policy"])
def index_stats():
    """Return statistics about the HR knowledge base index."""
    if _kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base not ready")
    return JSONResponse(content=_kb.index_stats())


@app.post("/policy/ingest", tags=["HR Policy"])
def ingest_document(body: IngestRequest):
    """
    Ingest an HR document into the knowledge base.

    Loads the document from the specified path, chunks it, embeds the
    chunks, and adds them to the vector store.  The index is persisted to
    disk after each ingest.
    """
    if _kb is None:
        raise HTTPException(status_code=503, detail="Knowledge base not ready")

    result = _kb.ingest_document(body.filepath, doc_type=body.doc_type)
    if not result.success:
        raise HTTPException(status_code=422, detail=result.message)

    return JSONResponse(content={
        "doc_id":       result.document_id,
        "filename":     result.filename,
        "chunks_added": result.chunks_added,
        "doc_type":     result.doc_type,
    })


# ── Onboarding endpoints ──────────────────────────────────────────────────────

@app.post("/onboarding/plans", tags=["Onboarding"])
def create_onboarding_plan(body: CreatePlanRequest):
    """
    Create a personalised onboarding plan for a new hire.

    Returns the full task list with due dates computed from the start date.
    Role-specific tasks are automatically added based on the role field.
    """
    if _onboarding is None:
        raise HTTPException(status_code=503, detail="Onboarding agent not ready")

    plan = _onboarding.create_plan(
        employee_id   = body.employee_id,
        employee_name = body.employee_name,
        role          = body.role,
        department    = body.department,
        start_date    = body.start_date,
        level         = body.level,
        manager_id    = body.manager_id,
    )
    return JSONResponse(content=plan.to_dict())


@app.get("/onboarding/{employee_id}", tags=["Onboarding"])
def get_onboarding_status(employee_id: str = Path(...)):
    """Get the full onboarding plan and task status for an employee."""
    if _onboarding is None:
        raise HTTPException(status_code=503, detail="Onboarding agent not ready")

    status = _onboarding.get_status(employee_id)
    if status is None:
        raise HTTPException(status_code=404, detail=f"No onboarding plan found for '{employee_id}'")

    return JSONResponse(content=status)


@app.get("/onboarding/{employee_id}/milestones", tags=["Onboarding"])
def get_milestone_summary(employee_id: str = Path(...)):
    """Return milestone-grouped completion summary for an employee."""
    if _onboarding is None:
        raise HTTPException(status_code=503, detail="Onboarding agent not ready")

    summary = _onboarding.milestone_summary(employee_id)
    if not summary:
        raise HTTPException(status_code=404, detail=f"No plan for '{employee_id}'")

    return JSONResponse(content=summary)


@app.get("/onboarding/{employee_id}/next-tasks", tags=["Onboarding"])
def get_next_tasks(employee_id: str = Path(...), n: int = Query(default=5, ge=1, le=20)):
    """Return the next N pending tasks for an employee."""
    if _onboarding is None:
        raise HTTPException(status_code=503, detail="Onboarding agent not ready")

    tasks = _onboarding.get_next_tasks(employee_id, n=n)
    return JSONResponse(content={"tasks": [t.to_dict() for t in tasks]})


@app.patch("/onboarding/{employee_id}/tasks/{task_id}/complete", tags=["Onboarding"])
def complete_task(
    employee_id: str = Path(...),
    task_id:     str = Path(...),
    body:        CompleteTaskRequest = CompleteTaskRequest(),
):
    """Mark an onboarding task as completed."""
    if _onboarding is None:
        raise HTTPException(status_code=503, detail="Onboarding agent not ready")

    task = _onboarding.complete_task(employee_id, task_id, notes=body.notes)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found for '{employee_id}'")

    return JSONResponse(content=task.to_dict())


@app.patch("/onboarding/{employee_id}/tasks/{task_id}/block", tags=["Onboarding"])
def block_task(
    employee_id: str = Path(...),
    task_id:     str = Path(...),
    body:        BlockTaskRequest = ...,
):
    """Flag an onboarding task as blocked."""
    if _onboarding is None:
        raise HTTPException(status_code=503, detail="Onboarding agent not ready")

    task = _onboarding.block_task(employee_id, task_id, reason=body.reason)
    if task is None:
        raise HTTPException(status_code=404, detail=f"Task '{task_id}' not found")

    return JSONResponse(content=task.to_dict())


@app.get("/onboarding/coordinator/dashboard", tags=["Onboarding"])
def coordinator_dashboard():
    """
    Coordinator dashboard showing all active onboarding plans.

    Surfaces employees with overdue or blocked tasks so the onboarding
    coordinator can take action.
    """
    if _onboarding is None:
        raise HTTPException(status_code=503, detail="Onboarding agent not ready")

    return JSONResponse(content=_onboarding.coordinator_dashboard())


# ── Benefits endpoints ────────────────────────────────────────────────────────

@app.get("/benefits/{employee_id}/pto", tags=["Benefits"])
def pto_balance(
    employee_id:     str  = Path(...),
    hire_date:       str  = Query(...),
    pto_used:        float = Query(default=0.0, ge=0),
    employment_type: str  = Query(default="full_time"),
):
    """
    Calculate an employee's current PTO balance.

    Accrual is computed from hire_date to today at the configured rate.
    Part-time employees are pro-rated based on employment_type.
    """
    if _benefits is None:
        raise HTTPException(status_code=503, detail="Benefits agent not ready")

    try:
        balance = _benefits.pto_balance(
            employee_id     = employee_id,
            hire_date       = hire_date,
            pto_used_days   = pto_used,
            employment_type = employment_type,
        )
        return JSONResponse(content=balance.to_dict())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc))


@app.get("/benefits/retirement-match", tags=["Benefits"])
def retirement_match(
    contribution_pct: float = Query(..., ge=0, le=100),
    annual_salary:    float = Query(default=100_000.0, gt=0),
):
    """
    Calculate employer 401(k) match for a given contribution rate.

    Returns the dollar amount of employer match, total combined contribution,
    and whether the employee is maximising the available match.
    """
    if _benefits is None:
        raise HTTPException(status_code=503, detail="Benefits agent not ready")

    result = _benefits.retirement_match(contribution_pct, annual_salary)
    return JSONResponse(content=result)


@app.get("/benefits/{employee_id}/summary", tags=["Benefits"])
def benefits_summary(
    employee_id:      str   = Path(...),
    hire_date:        str   = Query(...),
    pto_used:         float = Query(default=0.0, ge=0),
    plan_name:        str   = Query(default="PPO"),
    coverage_tier:    str   = Query(default="employee_only"),
    salary:           float = Query(default=100_000.0, gt=0),
    contribution_pct: float = Query(default=6.0, ge=0, le=100),
):
    """
    Generate a full personalised benefits summary.

    Combines PTO balance, retirement match, and health plan costs into
    one response.  Suitable for use in an employee self-service portal.
    """
    if _benefits is None:
        raise HTTPException(status_code=503, detail="Benefits agent not ready")

    summary = _benefits.benefits_summary(
        employee_id      = employee_id,
        hire_date        = hire_date,
        pto_used         = pto_used,
        plan_name        = plan_name,
        coverage_tier    = coverage_tier,
        salary           = salary,
        contribution_pct = contribution_pct,
    )
    return JSONResponse(content=summary)
