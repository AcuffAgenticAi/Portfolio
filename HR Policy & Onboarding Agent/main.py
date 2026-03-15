"""
================================================================================
main.py  —  HR Policy & Onboarding Agent  —  Platform Entry Point
================================================================================

Usage:
    python main.py                        # interactive HR chat demo
    python main.py --ingest               # ingest all HR documents into KB
    python main.py --onboard              # create a demo onboarding plan
    python main.py --benefits             # run benefits calculator demo
    python main.py --serve                # start FastAPI server (port 8002)
================================================================================
"""

import argparse
import json
import logging
import sys
from datetime import date, timedelta

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  [%(levelname)-8s]  %(name)s — %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("platform.main")
SEP    = "─" * 64


def run_ingest() -> None:
    """Ingest all HR documents into the knowledge base."""
    from data.generate_hr_documents       import generate_all
    from agents.knowledge_base_agent      import KnowledgeBaseAgent

    generate_all()
    kb = KnowledgeBaseAgent()
    results = kb.ingest_directory("knowledge_base/documents", doc_type="policy")
    print(f"\n{SEP}")
    print(f"  Ingested {len(results)} documents")
    for r in results:
        status = "OK" if r.success else "FAIL"
        print(f"  [{status}] {r.filename:<40} {r.chunks_added} chunks")
    print(f"  Total chunks in index: {kb.index_stats()['total_chunks']}")
    print(f"{SEP}\n")


def run_chat_demo() -> None:
    """Interactive HR policy Q&A demo."""
    from agents.knowledge_base_agent import KnowledgeBaseAgent
    from agents.hr_policy_agent      import HRPolicyAgent

    kb    = KnowledgeBaseAgent()
    if kb.index_stats()["total_chunks"] == 0:
        print("Knowledge base is empty. Run: python main.py --ingest")
        return

    agent = HRPolicyAgent(knowledge_base=kb)
    conv_id  = "demo-conv"
    emp_id   = "emp-demo"

    sample_questions = [
        "How many PTO days do I get per year?",
        "What is the employer 401k match?",
        "How do I request parental leave?",
        "What happens to my PTO if I don't use it all?",
    ]

    print(f"\n{SEP}")
    print("  HR POLICY ASSISTANT — DEMO")
    print(f"{SEP}\n")

    for question in sample_questions:
        print(f"  Q: {question}")
        response = agent.answer(question, employee_id=emp_id, conversation_id=conv_id)
        print(f"  A: {response.answer[:300]}...")
        if response.citations:
            print(f"     [Source: {response.citations[0]['filename']}]")
        if response.should_escalate:
            print(f"     ** ESCALATED: {response.escalation_reason[:80]} **")
        print()


def run_onboarding_demo() -> None:
    """Create and display a demo onboarding plan."""
    from agents.onboarding_agent import OnboardingAgent

    agent      = OnboardingAgent()
    start_date = (date.today() + timedelta(days=7)).isoformat()

    plan = agent.create_plan(
        employee_id   = "demo-eng-001",
        employee_name = "Alex Rivera",
        role          = "Senior Software Engineer",
        department    = "Engineering",
        start_date    = start_date,
        level         = "individual_contributor",
        manager_id    = "mgr-001",
    )

    print(f"\n{SEP}")
    print(f"  ONBOARDING PLAN  —  {plan.employee_name}")
    print(f"  Role: {plan.role}  |  Start: {plan.start_date}  |  Tasks: {len(plan.tasks)}")
    print(f"{SEP}\n")

    # Show first 10 tasks by due date
    for task in plan.tasks[:10]:
        req   = "* " if task.required else "  "
        print(f"  {req}{task.due_date}  [{task.owner:<14}]  {task.title}")

    if len(plan.tasks) > 10:
        print(f"\n  ... and {len(plan.tasks) - 10} more tasks.")

    # Milestone summary
    summary = agent.milestone_summary("demo-eng-001")
    print(f"\n  MILESTONES:")
    for milestone, data in summary.items():
        print(f"  {milestone:<20} {data['completed']}/{data['total']} tasks")
    print()


def run_benefits_demo() -> None:
    """Run benefits calculator demo."""
    from agents.benefits_agent import BenefitsAgent

    agent      = BenefitsAgent()
    hire_date  = date(date.today().year, 1, 1).isoformat()
    emp_id     = "demo-emp-001"

    print(f"\n{SEP}")
    print("  BENEFITS SUMMARY DEMO")
    print(f"{SEP}\n")

    pto = agent.pto_balance(emp_id, hire_date, pto_used_days=5.0)
    print(f"  PTO Balance:")
    print(f"    Accrued:    {pto.accrued_days:.1f} days")
    print(f"    Used:       {pto.used_days:.1f} days")
    print(f"    Remaining:  {pto.remaining_days:.1f} days")
    if pto.note:
        print(f"    Note: {pto.note}")

    print(f"\n  401(k) Match (salary $100k, contributing 6%):")
    match = agent.retirement_match(contribution_pct=6.0, annual_salary=100_000)
    print(f"    Your contribution:    ${match['employee_contribution_usd']:,.2f}/year")
    print(f"    Employer match:       ${match['employer_match_usd']:,.2f}/year")
    print(f"    Total:                ${match['total_annual_contribution']:,.2f}/year")
    print(f"    {match['tip']}")

    print(f"\n  PPO Health Plan (Employee + Family):")
    health = agent.health_plan_cost("PPO", "family")
    print(f"    Monthly premium (you): ${health['employee_monthly_cost']}")
    print(f"    Annual cost:           ${health['annual_employee_cost']:,.2f}")
    print(f"    Deductible:            ${health['deductible']:,}")
    print(f"    OOP Max:               ${health['out_of_pocket_max']:,}")
    print()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HR Policy & Onboarding Agent")
    group  = parser.add_mutually_exclusive_group()
    group.add_argument("--ingest",    action="store_true", help="Ingest HR documents into knowledge base")
    group.add_argument("--onboard",   action="store_true", help="Create a demo onboarding plan")
    group.add_argument("--benefits",  action="store_true", help="Run benefits calculator demo")
    group.add_argument("--serve",     action="store_true", help="Start FastAPI server on port 8002")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.ingest:
        run_ingest()
        return

    if args.onboard:
        run_onboarding_demo()
        return

    if args.benefits:
        run_benefits_demo()
        return

    if args.serve:
        import uvicorn
        from api.hr_api import app
        uvicorn.run(app, host="0.0.0.0", port=8002, log_level="info")
        return

    # Default: run all demos
    run_ingest()
    run_chat_demo()
    run_onboarding_demo()
    run_benefits_demo()


if __name__ == "__main__":
    main()
