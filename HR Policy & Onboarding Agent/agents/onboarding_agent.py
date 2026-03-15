"""
================================================================================
agents/onboarding_agent.py  —  New Hire Onboarding Agent
================================================================================

Purpose:
    Generates personalised onboarding checklists for new hires and tracks
    task completion through the onboarding lifecycle.

    Onboarding is one of the most chaotic processes in enterprise HR — tasks
    are scattered across IT, facilities, HR, and the hiring manager.  This
    agent centralises the checklist, personalises it by role/department/level,
    and tracks completion so nothing falls through the cracks.

Checklist personalisation:
    The base checklist is the same for every employee (system access, ID badge,
    benefits enrolment, compliance training).  The agent then adds role-specific
    tasks (e.g., engineers get "complete security training" and "set up dev
    environment"; managers get "complete manager training" and "meet skip-level").

Task lifecycle:
    PENDING → IN_PROGRESS → COMPLETED | SKIPPED | BLOCKED

    Blocked tasks surface to the onboarding coordinator automatically so they
    can resolve blockers (e.g., laptop not shipped, Workday not configured).

    At day 30, day 60, and day 90 the agent generates milestone check-in
    prompts to keep the extended onboarding on track.

Integration:
    The onboarding agent hands off specific tasks to the IT provisioning
    system (via mock API calls here) and the HRIS (Workday/SAP SuccessFactors)
    for benefits enrolment deadlines.

Usage:
    agent    = OnboardingAgent()
    plan     = agent.create_plan(employee_profile)
    status   = agent.get_status(employee_id)
    updated  = agent.complete_task(employee_id, task_id)
================================================================================
"""

from __future__ import annotations

import json
import logging
import os
import uuid
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Any, Dict, List, Optional

logger = logging.getLogger("platform.onboarding")

# ── Onboarding store path ─────────────────────────────────────────────────────
DEFAULT_STORE_PATH = "monitoring/onboarding_plans.json"

# ── Task status values ────────────────────────────────────────────────────────
STATUS_PENDING     = "pending"
STATUS_IN_PROGRESS = "in_progress"
STATUS_COMPLETED   = "completed"
STATUS_SKIPPED     = "skipped"
STATUS_BLOCKED     = "blocked"


# ─────────────────────────────────────────────────────────────────────────────
# Task template library
# ─────────────────────────────────────────────────────────────────────────────

# Base tasks every employee completes regardless of role
BASE_TASKS: List[Dict[str, Any]] = [
    # ── Pre-arrival (owner: HR) ───────────────────────────────────────────────
    {
        "id": "pre_01", "category": "pre_arrival", "owner": "HR",
        "title": "Complete I-9 employment eligibility verification",
        "description": "Submit identification documents via the HR portal before your start date.",
        "due_day": -3, "required": True, "milestone": "pre_arrival",
    },
    {
        "id": "pre_02", "category": "pre_arrival", "owner": "HR",
        "title": "Sign offer letter and employment agreement",
        "description": "Review and e-sign your offer letter in Workday.",
        "due_day": -5, "required": True, "milestone": "pre_arrival",
    },
    {
        "id": "pre_03", "category": "pre_arrival", "owner": "IT",
        "title": "Equipment order confirmed",
        "description": "IT will confirm your laptop and peripherals have been ordered.",
        "due_day": -7, "required": True, "milestone": "pre_arrival",
    },
    # ── Day 1 ─────────────────────────────────────────────────────────────────
    {
        "id": "d1_01", "category": "day_1", "owner": "Facilities",
        "title": "Collect ID badge from reception",
        "description": "Visit reception with your government-issued ID. Badge grants building access.",
        "due_day": 1, "required": True, "milestone": "day_1",
    },
    {
        "id": "d1_02", "category": "day_1", "owner": "IT",
        "title": "Set up corporate laptop",
        "description": "Complete laptop setup including disk encryption, VPN client, and SSO login.",
        "due_day": 1, "required": True, "milestone": "day_1",
    },
    {
        "id": "d1_03", "category": "day_1", "owner": "HR",
        "title": "Complete Day 1 orientation session",
        "description": "Attend the company orientation covering culture, values, and logistics.",
        "due_day": 1, "required": True, "milestone": "day_1",
    },
    {
        "id": "d1_04", "category": "day_1", "owner": "Manager",
        "title": "Meet your manager and team",
        "description": "1:1 with your manager + team introduction meeting.",
        "due_day": 1, "required": True, "milestone": "day_1",
    },
    # ── Week 1 ────────────────────────────────────────────────────────────────
    {
        "id": "w1_01", "category": "week_1", "owner": "IT",
        "title": "Configure corporate email and calendar",
        "description": "Set up Outlook/Gmail, configure signature, and sync to mobile.",
        "due_day": 3, "required": True, "milestone": "week_1",
    },
    {
        "id": "w1_02", "category": "week_1", "owner": "HR",
        "title": "Enrol in benefits (health, dental, vision)",
        "description": "Log in to Workday Benefits and complete enrolment. Deadline: 30 days.",
        "due_day": 5, "required": True, "milestone": "week_1",
    },
    {
        "id": "w1_03", "category": "week_1", "owner": "HR",
        "title": "Set up direct deposit in payroll system",
        "description": "Enter bank account details in Workday to receive your first paycheck.",
        "due_day": 5, "required": True, "milestone": "week_1",
    },
    {
        "id": "w1_04", "category": "week_1", "owner": "Compliance",
        "title": "Complete mandatory compliance training (Code of Conduct)",
        "description": "Complete the 60-minute Code of Conduct e-learning module in the LMS.",
        "due_day": 7, "required": True, "milestone": "week_1",
    },
    {
        "id": "w1_05", "category": "week_1", "owner": "Compliance",
        "title": "Complete mandatory data privacy training",
        "description": "Complete the GDPR/CCPA awareness module in the LMS.",
        "due_day": 7, "required": True, "milestone": "week_1",
    },
    # ── Month 1 ───────────────────────────────────────────────────────────────
    {
        "id": "m1_01", "category": "month_1", "owner": "HR",
        "title": "Complete 30-day check-in with HR",
        "description": "Schedule a 30-minute check-in with your HR Business Partner.",
        "due_day": 30, "required": True, "milestone": "30_day",
    },
    {
        "id": "m1_02", "category": "month_1", "owner": "Manager",
        "title": "Complete 30-day performance check-in with manager",
        "description": "Discuss initial goals, expectations, and feedback with your manager.",
        "due_day": 30, "required": True, "milestone": "30_day",
    },
    {
        "id": "m1_03", "category": "month_1", "owner": "Employee",
        "title": "Enrol in 401(k) plan",
        "description": "Log in to Fidelity/Vanguard portal to set contribution rate and investments.",
        "due_day": 30, "required": False, "milestone": "month_1",
    },
    # ── 90-day milestone ──────────────────────────────────────────────────────
    {
        "id": "q1_01", "category": "quarter_1", "owner": "Manager",
        "title": "Complete 90-day review",
        "description": "Formal performance review at the end of the initial 90-day period.",
        "due_day": 90, "required": True, "milestone": "90_day",
    },
    {
        "id": "q1_02", "category": "quarter_1", "owner": "Employee",
        "title": "Set annual performance goals in Workday",
        "description": "Enter and align performance goals with your manager in the HRIS.",
        "due_day": 90, "required": True, "milestone": "90_day",
    },
]

# Role-specific tasks appended to the base checklist
ROLE_TASKS: Dict[str, List[Dict[str, Any]]] = {
    "engineer": [
        {
            "id": "eng_01", "category": "role_specific", "owner": "IT",
            "title": "Complete security and secure coding training",
            "description": "Complete the developer security awareness module in the LMS.",
            "due_day": 7, "required": True, "milestone": "week_1",
        },
        {
            "id": "eng_02", "category": "role_specific", "owner": "Engineering",
            "title": "Set up local development environment",
            "description": "Follow the engineering onboarding runbook to configure your dev environment.",
            "due_day": 3, "required": True, "milestone": "week_1",
        },
        {
            "id": "eng_03", "category": "role_specific", "owner": "Engineering",
            "title": "Complete first code review participation",
            "description": "Review at least one PR and submit your first PR for review.",
            "due_day": 14, "required": False, "milestone": "week_2",
        },
    ],
    "manager": [
        {
            "id": "mgr_01", "category": "role_specific", "owner": "HR",
            "title": "Complete new manager orientation",
            "description": "Attend the new manager orientation covering performance management and HR processes.",
            "due_day": 14, "required": True, "milestone": "week_2",
        },
        {
            "id": "mgr_02", "category": "role_specific", "owner": "Manager",
            "title": "Set up 1:1s with all direct reports",
            "description": "Schedule recurring 1:1 meetings with each direct report.",
            "due_day": 5, "required": True, "milestone": "week_1",
        },
        {
            "id": "mgr_03", "category": "role_specific", "owner": "HR",
            "title": "Complete unconscious bias training",
            "description": "Mandatory training for all people managers in the LMS.",
            "due_day": 30, "required": True, "milestone": "30_day",
        },
    ],
    "sales": [
        {
            "id": "sales_01", "category": "role_specific", "owner": "Sales Enablement",
            "title": "Complete sales methodology certification",
            "description": "Complete the company sales methodology training (MEDDIC/Challenger/etc.).",
            "due_day": 14, "required": True, "milestone": "week_2",
        },
        {
            "id": "sales_02", "category": "role_specific", "owner": "IT",
            "title": "Set up CRM access (Salesforce)",
            "description": "IT will provision your Salesforce account and you will complete CRM training.",
            "due_day": 3, "required": True, "milestone": "week_1",
        },
    ],
    "finance": [
        {
            "id": "fin_01", "category": "role_specific", "owner": "Finance",
            "title": "Complete SOX compliance training",
            "description": "Mandatory Sarbanes-Oxley compliance training for all Finance employees.",
            "due_day": 7, "required": True, "milestone": "week_1",
        },
        {
            "id": "fin_02", "category": "role_specific", "owner": "IT",
            "title": "Provision financial systems access (NetSuite/SAP)",
            "description": "IT will provision access to the ERP system with appropriate role permissions.",
            "due_day": 3, "required": True, "milestone": "week_1",
        },
    ],
}


@dataclass
class OnboardingTask:
    """A single onboarding task with status tracking."""
    id:           str
    title:        str
    description:  str
    category:     str
    owner:        str
    due_date:     str   # ISO date string
    due_day:      int   # days relative to start date
    required:     bool
    milestone:    str
    status:       str   = STATUS_PENDING
    completed_at: str   = ""
    notes:        str   = ""
    blocked_reason: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id":            self.id,
            "title":         self.title,
            "description":   self.description,
            "category":      self.category,
            "owner":         self.owner,
            "due_date":      self.due_date,
            "required":      self.required,
            "milestone":     self.milestone,
            "status":        self.status,
            "completed_at":  self.completed_at,
            "notes":         self.notes,
            "blocked_reason": self.blocked_reason,
        }


@dataclass
class OnboardingPlan:
    """Complete onboarding plan for one employee."""
    employee_id:   str
    employee_name: str
    role:          str
    department:    str
    level:         str
    start_date:    str
    manager_id:    str
    plan_id:       str
    tasks:         List[OnboardingTask]
    created_at:    str

    def completion_pct(self) -> float:
        total     = len(self.tasks)
        completed = sum(1 for t in self.tasks if t.status == STATUS_COMPLETED)
        return round(completed / total * 100, 1) if total > 0 else 0.0

    def overdue_tasks(self) -> List[OnboardingTask]:
        today = date.today().isoformat()
        return [
            t for t in self.tasks
            if t.status == STATUS_PENDING and t.due_date < today
        ]

    def blocked_tasks(self) -> List[OnboardingTask]:
        return [t for t in self.tasks if t.status == STATUS_BLOCKED]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "plan_id":       self.plan_id,
            "employee_id":   self.employee_id,
            "employee_name": self.employee_name,
            "role":          self.role,
            "department":    self.department,
            "level":         self.level,
            "start_date":    self.start_date,
            "manager_id":    self.manager_id,
            "created_at":    self.created_at,
            "completion_pct": self.completion_pct(),
            "task_summary": {
                "total":       len(self.tasks),
                "completed":   sum(1 for t in self.tasks if t.status == STATUS_COMPLETED),
                "pending":     sum(1 for t in self.tasks if t.status == STATUS_PENDING),
                "in_progress": sum(1 for t in self.tasks if t.status == STATUS_IN_PROGRESS),
                "blocked":     sum(1 for t in self.tasks if t.status == STATUS_BLOCKED),
                "overdue":     len(self.overdue_tasks()),
            },
            "tasks": [t.to_dict() for t in self.tasks],
        }


class OnboardingAgent:
    """
    Creates and manages personalised onboarding plans for new hires.

    Attributes:
        store_path:  Path to persist onboarding plans as JSON.
        _plans:      In-memory dict of plan_id → OnboardingPlan.
    """

    def __init__(self, store_path: str = DEFAULT_STORE_PATH) -> None:
        self.store_path = store_path
        self._plans: Dict[str, OnboardingPlan] = {}
        self._load_plans()

    # ─────────────────────────────────────────────────────────────────────────
    # Plan creation
    # ─────────────────────────────────────────────────────────────────────────

    def create_plan(
        self,
        employee_id:   str,
        employee_name: str,
        role:          str,
        department:    str,
        start_date:    str,
        level:         str   = "individual_contributor",
        manager_id:    str   = "",
        custom_tasks:  Optional[List[Dict]] = None,
    ) -> OnboardingPlan:
        """
        Generate a personalised onboarding plan for a new hire.

        Args:
            employee_id:   Employee's unique ID.
            employee_name: Full name for display.
            role:          Job role (engineer, manager, sales, finance, other).
            department:    Department name.
            start_date:    ISO date string (YYYY-MM-DD) of first day.
            level:         Employment level (individual_contributor, manager, director+).
            manager_id:    Manager's employee ID.
            custom_tasks:  Optional list of additional task dicts to include.

        Returns:
            OnboardingPlan with all tasks, due dates calculated from start_date.
        """
        plan_id    = str(uuid.uuid4())[:12]
        start      = date.fromisoformat(start_date)

        # Build task list: base tasks + role-specific tasks
        role_key   = self._normalise_role(role)
        raw_tasks  = BASE_TASKS + ROLE_TASKS.get(role_key, [])

        # Add manager tasks if level is manager or above
        if level in ("manager", "director", "vp", "c_level"):
            for t in ROLE_TASKS.get("manager", []):
                if not any(r["id"] == t["id"] for r in raw_tasks):
                    raw_tasks.append(t)

        # Add custom tasks
        if custom_tasks:
            raw_tasks.extend(custom_tasks)

        # Deduplicate by task id
        seen    = set()
        unique  = []
        for t in raw_tasks:
            if t["id"] not in seen:
                seen.add(t["id"])
                unique.append(t)

        # Convert raw dicts to OnboardingTask dataclasses with computed due dates
        tasks = []
        for raw in unique:
            due    = start + timedelta(days=max(raw.get("due_day", 1), 1))
            # Pre-arrival tasks are before start_date
            if raw.get("due_day", 1) < 0:
                due = start + timedelta(days=raw["due_day"])

            tasks.append(OnboardingTask(
                id          = raw["id"],
                title       = raw["title"],
                description = raw["description"],
                category    = raw["category"],
                owner       = raw["owner"],
                due_date    = due.isoformat(),
                due_day     = raw.get("due_day", 1),
                required    = raw.get("required", True),
                milestone   = raw.get("milestone", "general"),
            ))

        # Sort by due_date
        tasks.sort(key=lambda t: t.due_date)

        plan = OnboardingPlan(
            employee_id   = employee_id,
            employee_name = employee_name,
            role          = role,
            department    = department,
            level         = level,
            start_date    = start_date,
            manager_id    = manager_id,
            plan_id       = plan_id,
            tasks         = tasks,
            created_at    = datetime.utcnow().isoformat(),
        )

        self._plans[employee_id] = plan
        self._save_plans()

        logger.info(
            "Onboarding plan created  employee=%s  role=%s  tasks=%d  plan_id=%s",
            employee_id, role, len(tasks), plan_id,
        )
        return plan

    # ─────────────────────────────────────────────────────────────────────────
    # Task management
    # ─────────────────────────────────────────────────────────────────────────

    def complete_task(
        self,
        employee_id: str,
        task_id:     str,
        notes:       str = "",
    ) -> Optional[OnboardingTask]:
        """
        Mark a task as completed.

        Args:
            employee_id:  Employee's ID.
            task_id:      Task ID to complete.
            notes:        Optional completion notes.

        Returns:
            Updated OnboardingTask, or None if not found.
        """
        plan = self._plans.get(employee_id)
        if not plan:
            logger.warning("No plan found for employee '%s'", employee_id)
            return None

        for task in plan.tasks:
            if task.id == task_id:
                task.status       = STATUS_COMPLETED
                task.completed_at = datetime.utcnow().isoformat()
                task.notes        = notes
                self._save_plans()
                logger.info(
                    "Task completed  employee=%s  task='%s'",
                    employee_id, task.title,
                )
                return task

        logger.warning("Task '%s' not found for employee '%s'", task_id, employee_id)
        return None

    def block_task(
        self,
        employee_id: str,
        task_id:     str,
        reason:      str,
    ) -> Optional[OnboardingTask]:
        """Flag a task as blocked with a reason (surfaces to coordinator)."""
        plan = self._plans.get(employee_id)
        if not plan:
            return None
        for task in plan.tasks:
            if task.id == task_id:
                task.status         = STATUS_BLOCKED
                task.blocked_reason = reason
                self._save_plans()
                logger.warning(
                    "Task BLOCKED  employee=%s  task='%s'  reason='%s'",
                    employee_id, task.title, reason,
                )
                return task
        return None

    def get_status(self, employee_id: str) -> Optional[Dict[str, Any]]:
        """Return the full onboarding status for an employee."""
        plan = self._plans.get(employee_id)
        if not plan:
            return None
        return plan.to_dict()

    def get_next_tasks(
        self,
        employee_id: str,
        n:           int = 5,
    ) -> List[OnboardingTask]:
        """Return the next N pending tasks sorted by due date."""
        plan = self._plans.get(employee_id)
        if not plan:
            return []
        pending = [t for t in plan.tasks if t.status == STATUS_PENDING]
        return sorted(pending, key=lambda t: t.due_date)[:n]

    def milestone_summary(self, employee_id: str) -> Dict[str, Any]:
        """
        Return a milestone-grouped completion summary.

        Groups tasks by milestone (pre_arrival, day_1, week_1, 30_day, 90_day)
        and shows completion percentage per group.
        """
        plan = self._plans.get(employee_id)
        if not plan:
            return {}

        milestones: Dict[str, Dict] = {}
        for task in plan.tasks:
            m = task.milestone
            if m not in milestones:
                milestones[m] = {"total": 0, "completed": 0, "tasks": []}
            milestones[m]["total"] += 1
            if task.status == STATUS_COMPLETED:
                milestones[m]["completed"] += 1
            milestones[m]["tasks"].append(task.to_dict())

        # Compute completion pct per milestone
        for m_data in milestones.values():
            t = m_data["total"]
            c = m_data["completed"]
            m_data["completion_pct"] = round(c / t * 100, 1) if t > 0 else 0.0

        return milestones

    def coordinator_dashboard(self) -> Dict[str, Any]:
        """
        Return a dashboard view for the onboarding coordinator.

        Shows all active onboarding plans with overdue/blocked task counts.
        """
        active = []
        for emp_id, plan in self._plans.items():
            overdue = plan.overdue_tasks()
            blocked = plan.blocked_tasks()
            active.append({
                "employee_id":   emp_id,
                "employee_name": plan.employee_name,
                "role":          plan.role,
                "start_date":    plan.start_date,
                "completion_pct": plan.completion_pct(),
                "overdue_count": len(overdue),
                "blocked_count": len(blocked),
                "needs_attention": len(overdue) > 0 or len(blocked) > 0,
                "overdue_tasks": [t.title for t in overdue[:3]],
                "blocked_tasks": [
                    {"title": t.title, "reason": t.blocked_reason} for t in blocked[:3]
                ],
            })
        return {
            "total_employees_onboarding": len(active),
            "employees_needing_attention": sum(1 for e in active if e["needs_attention"]),
            "plans": sorted(active, key=lambda e: -int(e["needs_attention"])),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Persistence
    # ─────────────────────────────────────────────────────────────────────────

    def _save_plans(self) -> None:
        os.makedirs(os.path.dirname(self.store_path) or ".", exist_ok=True)
        serialised = {emp_id: plan.to_dict() for emp_id, plan in self._plans.items()}
        with open(self.store_path, "w", encoding="utf-8") as fh:
            json.dump(serialised, fh, indent=2)

    def _load_plans(self) -> None:
        if not os.path.isfile(self.store_path):
            return
        try:
            with open(self.store_path, "r", encoding="utf-8") as fh:
                raw = json.load(fh)
            for emp_id, plan_dict in raw.items():
                tasks = [
                    OnboardingTask(
                        id            = t["id"],
                        title         = t["title"],
                        description   = t["description"],
                        category      = t["category"],
                        owner         = t["owner"],
                        due_date      = t["due_date"],
                        due_day       = t.get("due_day", 1),
                        required      = t.get("required", True),
                        milestone     = t.get("milestone", "general"),
                        status        = t.get("status", STATUS_PENDING),
                        completed_at  = t.get("completed_at", ""),
                        notes         = t.get("notes", ""),
                        blocked_reason = t.get("blocked_reason", ""),
                    )
                    for t in plan_dict.get("tasks", [])
                ]
                self._plans[emp_id] = OnboardingPlan(
                    employee_id   = plan_dict["employee_id"],
                    employee_name = plan_dict["employee_name"],
                    role          = plan_dict["role"],
                    department    = plan_dict["department"],
                    level         = plan_dict.get("level", "individual_contributor"),
                    start_date    = plan_dict["start_date"],
                    manager_id    = plan_dict.get("manager_id", ""),
                    plan_id       = plan_dict["plan_id"],
                    tasks         = tasks,
                    created_at    = plan_dict["created_at"],
                )
            logger.info("Loaded %d onboarding plans from '%s'", len(self._plans), self.store_path)
        except Exception as exc:  # noqa: BLE001
            logger.error("Failed to load onboarding plans: %s", exc)

    @staticmethod
    def _normalise_role(role: str) -> str:
        """Map free-text role to a task template key."""
        r = role.lower()
        if any(k in r for k in ("engineer", "developer", "swe", "sde", "software")):
            return "engineer"
        if any(k in r for k in ("manager", "director", "vp", "head of", "lead")):
            return "manager"
        if any(k in r for k in ("sales", "account executive", "ae", "sdr", "bdr")):
            return "sales"
        if any(k in r for k in ("finance", "accounting", "controller", "analyst")):
            return "finance"
        return "other"
