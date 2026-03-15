"""
================================================================================
agents/benefits_agent.py  —  Benefits & PTO Calculator Agent
================================================================================

Purpose:
    Computes personalised benefits summaries and PTO balances for employees.
    Answers the most common HR questions that require computation, not just
    policy lookup:

        "How many vacation days do I have left this year?"
        "What is the employer match for my 401(k) contribution?"
        "If I work 32 hours, what is my pro-rated PTO?"
        "When does my COBRA coverage end?"
        "How much is my life insurance benefit?"

    This separates the structured calculation logic from the RAG-based
    policy Q&A — calculations run deterministically without LLM inference.

Benefits catalogue (configurable via JSON):
    PTO Policy:     accrual rate, carryover cap, payout on termination
    Health Plans:   premium, deductible, OOP max per tier (employee/+1/family)
    401(k):         employer match formula, vesting schedule, contribution limits
    Life Insurance: coverage multiple, supplemental options
    COBRA:          duration, premium calculation
    Equity:         vesting cliff, schedule, acceleration events

Usage:
    agent   = BenefitsAgent()
    balance = agent.pto_balance(employee_id, hire_date, pto_used)
    match   = agent.retirement_match(contribution_pct=6.0)
    summary = agent.benefits_summary(employee_id, plan_tier="family")
================================================================================
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass
from datetime import date, datetime
from typing import Any, Dict, List, Optional

logger = logging.getLogger("platform.benefits")

# ── Default benefits catalogue path ──────────────────────────────────────────
DEFAULT_CATALOGUE_PATH = "data/benefits_catalogue.json"


@dataclass
class PTOBalance:
    """PTO balance calculation result."""
    employee_id:      str
    accrued_days:     float
    used_days:        float
    remaining_days:   float
    carryover_cap:    float
    days_until_reset: int
    accrual_rate:     float    # days per month
    note:             str = ""

    def to_dict(self) -> Dict[str, Any]:
        return {
            "employee_id":      self.employee_id,
            "accrued_days":     round(self.accrued_days, 1),
            "used_days":        round(self.used_days, 1),
            "remaining_days":   round(self.remaining_days, 1),
            "carryover_cap":    self.carryover_cap,
            "days_until_reset": self.days_until_reset,
            "accrual_rate":     self.accrual_rate,
            "note":             self.note,
        }


class BenefitsAgent:
    """
    Computes personalised benefits information from structured data.

    Attributes:
        catalogue:  Benefits catalogue dict loaded from JSON.
    """

    def __init__(self, catalogue_path: str = DEFAULT_CATALOGUE_PATH) -> None:
        self.catalogue = self._load_catalogue(catalogue_path)

    # ─────────────────────────────────────────────────────────────────────────
    # PTO calculations
    # ─────────────────────────────────────────────────────────────────────────

    def pto_balance(
        self,
        employee_id:    str,
        hire_date:      str,     # ISO date: YYYY-MM-DD
        pto_used_days:  float,
        employment_type: str = "full_time",
        as_of_date:     Optional[str] = None,
    ) -> PTOBalance:
        """
        Calculate an employee's current PTO balance.

        Accrual is calculated from hire_date to as_of_date (default: today)
        at the rate configured in the benefits catalogue.  Part-time employees
        accrue pro-rated based on their FTE percentage.

        Args:
            employee_id:     Employee identifier.
            hire_date:       ISO date of hire.
            pto_used_days:   Days already taken this accrual year.
            employment_type: "full_time" | "part_time_80" | "part_time_60" | etc.
            as_of_date:      Calculate as of this date (default: today).

        Returns:
            PTOBalance dataclass.
        """
        pto_config   = self.catalogue.get("pto", {})
        accrual_rate = float(pto_config.get("accrual_rate_days_per_month", 1.75))
        carryover    = float(pto_config.get("carryover_cap_days", 5.0))
        max_annual   = float(pto_config.get("max_annual_days", 21.0))

        # Pro-rate for part-time
        fte_map = {"full_time": 1.0, "part_time_80": 0.8, "part_time_60": 0.6, "part_time_50": 0.5}
        fte_pct = fte_map.get(employment_type, 1.0)
        effective_rate = accrual_rate * fte_pct

        # Compute months employed this year
        today        = date.fromisoformat(as_of_date) if as_of_date else date.today()
        hire         = date.fromisoformat(hire_date)
        year_start   = date(today.year, 1, 1)
        accrual_from = max(hire, year_start)
        months       = (today.year - accrual_from.year) * 12 + (today.month - accrual_from.month)
        months       = max(months, 0)

        accrued   = min(effective_rate * months, max_annual * fte_pct)
        remaining = max(accrued - pto_used_days, 0.0)

        # Days until next year reset
        next_year_start  = date(today.year + 1, 1, 1)
        days_until_reset = (next_year_start - today).days

        note = ""
        if remaining > carryover and days_until_reset < 60:
            forfeited = remaining - carryover
            note = (
                f"Warning: {forfeited:.1f} days will be forfeited on {next_year_start}. "
                f"Only {carryover:.0f} days carry over."
            )

        return PTOBalance(
            employee_id      = employee_id,
            accrued_days     = accrued,
            used_days        = pto_used_days,
            remaining_days   = remaining,
            carryover_cap    = carryover,
            days_until_reset = days_until_reset,
            accrual_rate     = effective_rate,
            note             = note,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # 401(k) calculations
    # ─────────────────────────────────────────────────────────────────────────

    def retirement_match(
        self,
        contribution_pct: float,
        annual_salary:    float = 100_000.0,
    ) -> Dict[str, Any]:
        """
        Calculate the employer 401(k) match for a given employee contribution.

        The match formula is loaded from the benefits catalogue.
        Standard formula: 100% match on first 3%, 50% match on next 2%.

        Args:
            contribution_pct: Employee's contribution as % of salary.
            annual_salary:    Employee's annual salary.

        Returns:
            Dict with employee_contribution, employer_match, total_contribution.
        """
        match_config  = self.catalogue.get("retirement_401k", {})
        match_tiers   = match_config.get("match_tiers", [
            {"up_to_pct": 3.0, "match_pct": 100.0},
            {"up_to_pct": 2.0, "match_pct": 50.0},
        ])
        irs_limit     = float(match_config.get("irs_contribution_limit_2026", 23_500.0))
        catch_up      = float(match_config.get("catch_up_contribution_50plus", 7_500.0))

        emp_contribution = min(annual_salary * contribution_pct / 100, irs_limit)
        employer_match   = 0.0
        remaining_pct    = contribution_pct

        for tier in match_tiers:
            tier_pct  = float(tier["up_to_pct"])
            match_pct = float(tier["match_pct"]) / 100
            applied   = min(remaining_pct, tier_pct)
            employer_match   += annual_salary * applied / 100 * match_pct
            remaining_pct    -= applied
            if remaining_pct <= 0:
                break

        return {
            "employee_contribution_pct":  contribution_pct,
            "employee_contribution_usd":  round(emp_contribution, 2),
            "employer_match_usd":         round(employer_match, 2),
            "total_annual_contribution":  round(emp_contribution + employer_match, 2),
            "irs_limit_2026":             irs_limit,
            "effective_match_rate":       round(employer_match / max(emp_contribution, 1) * 100, 1),
            "tip": (
                f"To maximise employer match, contribute at least "
                f"{sum(t['up_to_pct'] for t in match_tiers):.0f}% of salary."
            ) if contribution_pct < sum(t["up_to_pct"] for t in match_tiers) else
            "You are maximising your employer match.",
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Health insurance costs
    # ─────────────────────────────────────────────────────────────────────────

    def health_plan_cost(
        self,
        plan_name:       str,
        coverage_tier:   str = "employee_only",
    ) -> Dict[str, Any]:
        """
        Return monthly premium, deductible, and OOP max for a health plan tier.

        Args:
            plan_name:     Plan name from the catalogue (e.g., "HMO", "PPO", "HDHP").
            coverage_tier: "employee_only" | "employee_plus_one" | "family".
        """
        plans = self.catalogue.get("health_plans", {})
        plan  = plans.get(plan_name)
        if not plan:
            return {
                "error": f"Plan '{plan_name}' not found. Available: {list(plans.keys())}"
            }

        tiers     = plan.get("tiers", {})
        tier_data = tiers.get(coverage_tier, {})
        employer_premium = float(tier_data.get("employer_monthly_premium", 0))
        employee_premium = float(tier_data.get("employee_monthly_premium", 0))

        return {
            "plan_name":               plan_name,
            "coverage_tier":           coverage_tier,
            "employee_monthly_cost":   employee_premium,
            "employer_monthly_cost":   employer_premium,
            "annual_employee_cost":    round(employee_premium * 12, 2),
            "deductible":              tier_data.get("deductible"),
            "out_of_pocket_max":       tier_data.get("oop_max"),
            "in_network_copay":        tier_data.get("copay"),
            "plan_type":               plan.get("type", ""),
            "hsa_eligible":            plan.get("hsa_eligible", False),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Full benefits summary
    # ─────────────────────────────────────────────────────────────────────────

    def benefits_summary(
        self,
        employee_id:   str,
        hire_date:     str,
        pto_used:      float = 0.0,
        plan_name:     str   = "PPO",
        coverage_tier: str   = "employee_only",
        salary:        float = 100_000.0,
        contribution_pct: float = 6.0,
    ) -> Dict[str, Any]:
        """
        Generate a complete benefits summary for an employee.

        Args:
            employee_id:      Employee identifier.
            hire_date:        ISO date of hire.
            pto_used:         PTO days already taken this year.
            plan_name:        Health plan name.
            coverage_tier:    Health coverage tier.
            salary:           Annual salary for 401(k) calculations.
            contribution_pct: 401(k) contribution percentage.

        Returns:
            Dict with pto, retirement, health, and miscellaneous benefits.
        """
        return {
            "employee_id":  employee_id,
            "generated_at": datetime.utcnow().isoformat(),
            "pto":          self.pto_balance(employee_id, hire_date, pto_used).to_dict(),
            "retirement":   self.retirement_match(contribution_pct, salary),
            "health":       self.health_plan_cost(plan_name, coverage_tier),
            "misc_benefits": self.catalogue.get("misc_benefits", {}),
        }

    # ─────────────────────────────────────────────────────────────────────────
    # Catalogue loading
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _load_catalogue(path: str) -> Dict[str, Any]:
        """Load the benefits catalogue from JSON, returning defaults if absent."""
        if os.path.isfile(path):
            try:
                with open(path, "r", encoding="utf-8") as fh:
                    catalogue = json.load(fh)
                logger.info("Benefits catalogue loaded from '%s'", path)
                return catalogue
            except Exception as exc:  # noqa: BLE001
                logger.error("Failed to load benefits catalogue: %s", exc)

        logger.info("Using default benefits catalogue")
        return {
            "pto": {
                "accrual_rate_days_per_month": 1.75,
                "max_annual_days":             21.0,
                "carryover_cap_days":          5.0,
                "payout_on_termination":       True,
            },
            "retirement_401k": {
                "match_tiers": [
                    {"up_to_pct": 3.0, "match_pct": 100.0},
                    {"up_to_pct": 2.0, "match_pct": 50.0},
                ],
                "vesting_schedule":               "4-year graded (25% per year)",
                "irs_contribution_limit_2026":    23_500,
                "catch_up_contribution_50plus":   7_500,
            },
            "health_plans": {
                "PPO": {
                    "type": "PPO", "hsa_eligible": False,
                    "tiers": {
                        "employee_only":      {"employer_monthly_premium": 450, "employee_monthly_premium": 120, "deductible": 1000, "oop_max": 4000, "copay": 25},
                        "employee_plus_one":  {"employer_monthly_premium": 850, "employee_monthly_premium": 280, "deductible": 2000, "oop_max": 8000, "copay": 25},
                        "family":             {"employer_monthly_premium": 1200, "employee_monthly_premium": 400, "deductible": 3000, "oop_max": 12000, "copay": 25},
                    },
                },
                "HDHP": {
                    "type": "HDHP", "hsa_eligible": True,
                    "tiers": {
                        "employee_only":      {"employer_monthly_premium": 500, "employee_monthly_premium": 60, "deductible": 2800, "oop_max": 5000, "copay": 0},
                        "employee_plus_one":  {"employer_monthly_premium": 950, "employee_monthly_premium": 140, "deductible": 5600, "oop_max": 10000, "copay": 0},
                        "family":             {"employer_monthly_premium": 1350, "employee_monthly_premium": 200, "deductible": 5600, "oop_max": 14000, "copay": 0},
                    },
                },
                "HMO": {
                    "type": "HMO", "hsa_eligible": False,
                    "tiers": {
                        "employee_only":      {"employer_monthly_premium": 480, "employee_monthly_premium": 80, "deductible": 500, "oop_max": 3000, "copay": 20},
                        "employee_plus_one":  {"employer_monthly_premium": 900, "employee_monthly_premium": 200, "deductible": 1000, "oop_max": 6000, "copay": 20},
                        "family":             {"employer_monthly_premium": 1280, "employee_monthly_premium": 320, "deductible": 1500, "oop_max": 9000, "copay": 20},
                    },
                },
            },
            "misc_benefits": {
                "life_insurance":        "2x annual salary (employer paid)",
                "short_term_disability": "60% of salary, up to 12 weeks",
                "long_term_disability":  "60% of salary, 90-day elimination period",
                "eap":                   "6 free counselling sessions per year",
                "gym_subsidy":           "$50/month reimbursement",
                "commuter_benefit":      "$280/month pre-tax transit/parking",
                "learning_budget":       "$2,000/year for professional development",
                "work_from_home":        "$500 annual home office stipend",
            },
        }
