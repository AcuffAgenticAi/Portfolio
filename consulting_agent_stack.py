#!/usr/bin/env python3
"""
╔══════════════════════════════════════════════════════════════╗
║        AI CONSULTING AGENT STACK  v1.0                       ║
║   7 Specialist Agents · One-Person Agency Powerhouse         ║
║   Powered by Claude · Built for $25k–$80k/mo Operations      ║
╚══════════════════════════════════════════════════════════════╝

AGENTS:
  1. Lead Generation      — Finds high-value prospects
  2. Outreach             — Personalized email & LinkedIn campaigns
  3. Discovery/Diagnosis  — Deep business analysis & AI opportunity map
  4. Proposal Generation  — Professional proposals with pricing
  5. Market Intelligence  — Competitor & trend monitoring
  6. Automation Builder   — Generates working Python & workflow scripts
  7. Client Reporting     — KPI dashboards & growth reports

BONUS:
  8. AI Business Consultant — The flagship diagnostic agent

USAGE:
  python consulting_agent_stack.py                        # interactive menu
  python consulting_agent_stack.py --agent all            # run full pipeline
  python consulting_agent_stack.py --agent discovery \\
      --company "Acme Corp" --industry "Retail" \\
      --employees "50" --services "E-commerce"
  python consulting_agent_stack.py --agent ai_consultant \\
      --company "GreenLeaf" --industry "Logistics" \\
      --employees "120" --services "Last-mile delivery"
"""

import anthropic
import argparse
import sys
import textwrap
import threading
import time
from datetime import datetime
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────
MODEL      = "claude-sonnet-4-20250514"
MAX_TOKENS = 2048
OUTPUT_DIR = Path("consulting_output")

client = anthropic.Anthropic()

# ── ANSI COLORS ───────────────────────────────────────────────────────────────
R  = "\033[0m";   BD = "\033[1m";   DM = "\033[2m"
RD = "\033[31m";  GR = "\033[32m";  YL = "\033[33m"
BL = "\033[34m";  MG = "\033[35m";  CY = "\033[36m"
WH = "\033[97m";  OR = "\033[38;5;208m"

# ── AGENT REGISTRY ────────────────────────────────────────────────────────────
AGENTS = [
    {
        "id": "lead_gen",
        "name": "Lead Generation Agent",
        "emoji": "🔍",
        "color": CY,
        "tagline": "Finds high-value prospects automatically",
        "inputs": {
            "target_industry": "Industry to target (e.g. SaaS, Healthcare, Retail)",
            "company_size":    "Company size range (e.g. 10–200 employees)",
            "geography":       "Target geography (e.g. US, Remote, Southeast)",
        },
        "system": """You are an expert B2B Lead Generation Specialist for an AI consulting agency.
Given a target market, produce a detailed, actionable prospect analysis.

Respond with this EXACT structure:

## IDEAL CUSTOMER PROFILE
Describe company size, industry signals, budget indicators, and tech maturity.

## TOP 10 PROSPECT COMPANY PROFILES
For each profile provide:
- Company type & size
- Industry vertical
- Key pain points (2–3 bullets)
- Estimated AI opportunity: High / Med / Low
- Best contact role to target
- Recommended outreach angle

## LEAD SCORING CRITERIA
List 5–6 weighted criteria (e.g. company size 20%, tech adoption 30%)

## PROSPECT RESEARCH CHECKLIST
What to verify before reaching out (7–10 items)

## RECOMMENDED DIRECTORIES & DATA SOURCES
Where to find these companies (Apollo, LinkedIn Sales Nav, G2, etc.)

Be specific and actionable. No fluff.""",
    },
    {
        "id": "outreach",
        "name": "Outreach Agent",
        "emoji": "📧",
        "color": BL,
        "tagline": "Personalized multi-touch prospect communication",
        "inputs": {
            "prospect_company": "Target company name",
            "prospect_role":    "Target contact role (e.g. COO, Head of Ops)",
            "pain_point":       "Primary pain point to address",
            "your_service":     "Your service offering (e.g. AI workflow automation)",
        },
        "system": """You are an expert B2B Outreach Strategist for an AI consulting firm.
Create a complete, ready-to-use outreach campaign — non-salesy, value-first.

Respond with this EXACT structure:

## COLD EMAIL SEQUENCE (5 EMAILS)
For each email:
- Subject line + A/B variant
- Full body (150–200 words max)
- Send timing (Day X)
- Goal of this touch

## LINKEDIN MESSAGE SEQUENCE (3 MESSAGES)
- Connection request note (under 300 chars)
- First message after connection
- Value-add follow-up (share insight, not pitch)

## FOLLOW-UP CADENCE
Day 1 / Day 3 / Day 7 / Day 14 / Day 21 — what action to take each day

## PERSONALIZATION VARIABLES
List what to research per prospect before sending (5–8 items)

## CALL-TO-ACTION OPTIONS (ranked by conversion likelihood)
1. Low-friction CTA
2. Medium CTA
3. High-intent CTA

Write conversationally. Lead with their problem, not your solution.""",
    },
    {
        "id": "discovery",
        "name": "Discovery & Diagnosis Agent",
        "emoji": "🩺",
        "color": GR,
        "tagline": "Deep business analysis & AI opportunity mapping",
        "inputs": {
            "company_name":  "Company name",
            "industry":      "Industry / vertical",
            "employees":     "Number of employees",
            "services":      "Products or services offered",
            "revenue_range": "Annual revenue range (optional)",
        },
        "system": """You are a senior AI Business Consultant conducting a deep discovery analysis.
Produce a thorough diagnostic report with quantified findings.

Respond with this EXACT structure:

## EXECUTIVE SUMMARY
3–4 sentence overview of the company's AI readiness and biggest opportunities.

## TOP 10 OPERATIONAL PAIN POINTS
For each pain point:
- Description
- Business impact (time wasted / cost / lost revenue)
- Root cause
- Urgency: Critical / High / Medium

## AI AUTOMATION OPPORTUNITY MAP
| Process | Automation Score (1–10) | Est. Annual ROI | Complexity | Priority |
List 8–10 processes in this table.

## QUICK WINS (0–30 DAYS)
3 automations implementable immediately with minimal investment.

## MEDIUM-TERM ROADMAP (30–90 DAYS)
3–4 initiatives with expected outcomes.

## STRATEGIC INITIATIVES (90–180 DAYS)
2–3 transformational projects.

## ROI SUMMARY
- Hours saved per week: X
- Annual cost savings: $X–$X
- Revenue impact potential: $X
- Payback period: X months

## RECOMMENDED AI AGENTS TO BUILD
List 5–7 specific agents with their purpose and estimated build time.

Be quantitative. Use realistic, defensible numbers.""",
    },
    {
        "id": "proposal",
        "name": "Proposal Generation Agent",
        "emoji": "📋",
        "color": YL,
        "tagline": "Professional consulting proposals in minutes",
        "inputs": {
            "company_name":    "Client company name",
            "pain_points":     "Key pain points identified (from discovery)",
            "budget_range":    "Estimated client budget range",
            "timeline":        "Desired project timeline",
            "discovery_notes": "Additional context from discovery call",
        },
        "system": """You are an expert Consulting Proposal Writer for an AI automation firm.
Create a polished, persuasive proposal that closes deals.

Respond with this EXACT structure:

## EXECUTIVE SUMMARY
Compelling overview (problem → solution → investment → ROI). Max 200 words.

## OUR UNDERSTANDING OF YOUR SITUATION
Show you deeply understand their specific challenges and goals.

## PROPOSED SOLUTION

### Phase 1: Foundation — Weeks 1–2
Deliverables, activities, outcomes.

### Phase 2: Build — Weeks 3–6
Deliverables, activities, outcomes.

### Phase 3: Scale — Weeks 7–12
Deliverables, activities, outcomes.

## INVESTMENT OPTIONS

### Starter — $X,XXX (one-time)
Scope, deliverables, timeline.

### Growth — $X,XXX (one-time)  ← RECOMMENDED
Scope, deliverables, timeline.

### Retainer — $X,XXX/month
Scope, deliverables, ongoing support.

## EXPECTED ROI & OUTCOMES
Quantified results tied to their specific situation. Include timeline to ROI.

## WHY US
3–4 concrete differentiators. No generic claims.

## NEXT STEPS
Clear 3-step action plan to move forward.

## TERMS SUMMARY
Payment terms, IP ownership, confidentiality, revision policy.

Justify every price point with ROI. Write to close.""",
    },
    {
        "id": "market_intel",
        "name": "Market Intelligence Agent",
        "emoji": "📡",
        "color": MG,
        "tagline": "Continuous industry & competitor monitoring",
        "inputs": {
            "industry":      "Target industry to analyze",
            "service_type":  "Type of consulting service (e.g. AI automation)",
            "geography":     "Market geography",
            "company_stage": "Your agency stage (early / growing / established)",
        },
        "system": """You are a Market Intelligence Analyst specializing in AI consulting markets.
Produce a comprehensive market intelligence brief.

Respond with this EXACT structure:

## INDUSTRY OVERVIEW
Current state, growth trajectory, key driving forces, and disruption signals.

## COMPETITOR LANDSCAPE
For 5–6 competitors:
- Company name & positioning
- Pricing signals
- Technology stack
- Marketing approach
- Strengths / Weaknesses
- Gap you can exploit

## AI TECHNOLOGY ADOPTION TRENDS
What tools are being adopted, at what rate, and what that means for you.

## PRICING INTELLIGENCE
Market rates for comparable consulting services (ranges by service type).

## UNDERSERVED OPPORTUNITIES
3–5 niches or service gaps no one is owning yet.

## THREAT ASSESSMENT
Top risks: commoditization, new entrants, tech changes, economic factors.

## POSITIONING RECOMMENDATIONS
How to differentiate and win in this landscape. Be specific.

## WEEKLY MONITORING CHECKLIST
What to track every week to stay ahead (10–12 items with sources).

Be analytical. Reference real trends and real tools.""",
    },
    {
        "id": "automation_builder",
        "name": "Automation Builder Agent",
        "emoji": "⚙️",
        "color": OR,
        "tagline": "Generates working Python automations & AI agent templates",
        "inputs": {
            "process_name":   "Name of the process to automate",
            "process_desc":   "Describe what the process does today (manually)",
            "inputs_outputs": "What goes in, what should come out",
            "tools_used":     "Current tools involved (e.g. Gmail, Sheets, CRM)",
        },
        "system": """You are a senior Automation Engineer and AI systems architect.
Given a business process, produce complete, working code and workflow designs.

Respond with this EXACT structure:

## AUTOMATION ARCHITECTURE
ASCII diagram of the system: triggers → processing → outputs → storage.

## PYTHON AUTOMATION SCRIPT
```python
# Full, runnable script with:
# - All imports
# - Configuration section (use env vars for secrets)
# - Core logic with try/except error handling
# - Logging with timestamps
# - Main execution block
# - Detailed inline comments
```

## ZAPIER / MAKE WORKFLOW DESIGN
Step-by-step workflow: trigger → filters → actions → error paths.
Label each step with app name and action type.

## REUSABLE AI AGENT TEMPLATE
```python
# Clean agent class with:
# - __init__ with system prompt
# - run(inputs) method
# - Claude API call with error handling
# - Structured output parser
```

## SETUP INSTRUCTIONS
Step-by-step to get running (pip installs, env vars, config).

## TIME SAVINGS ESTIMATE
Before vs. after: hours/week saved, error rate reduction.

Write production-quality code. Error handling and logging throughout.""",
    },
    {
        "id": "client_reporting",
        "name": "Client Reporting Agent",
        "emoji": "📊",
        "color": RD,
        "tagline": "Automated KPI dashboards & growth reports",
        "inputs": {
            "client_name":   "Client company name",
            "project_type":  "Type of automation project delivered",
            "metrics":       "Key metrics being tracked",
            "period":        "Reporting period (e.g. Month 1, Q1 2025)",
            "results_data":  "Results data you have (rough estimates OK)",
        },
        "system": """You are a Client Success and Reporting Specialist for an AI consulting firm.
Produce a polished client report that demonstrates value and justifies retainer continuation.

Respond with this EXACT structure:

## EXECUTIVE SUMMARY
What was delivered, what moved, what's coming. Lead with wins. Max 150 words.

## KPI DASHBOARD
| Metric | Baseline | This Period | Change | Target |
List all tracked metrics in this table.

## WINS THIS PERIOD
3–5 specific, quantified wins with impact descriptions.

## AUTOMATION PERFORMANCE
For each automation running:
- Runs executed
- Success rate
- Time saved
- Errors caught / resolved

## GROWTH TREND DATA
Chart-ready data table showing trajectory over time.

## ISSUES & RESOLUTIONS
Problems encountered and how they were handled.

## NEXT PERIOD RECOMMENDATIONS
3–5 specific actions to drive more results next period.

## ROI SUMMARY
Total investment to date vs. total value generated.
Make the math undeniable.

Write to make the client feel the investment was worth every dollar.""",
    },
]

# ── FLAGSHIP DIAGNOSTIC AGENT ─────────────────────────────────────────────────
FLAGSHIP_AGENT = {
    "id": "ai_consultant",
    "name": "AI Business Consultant",
    "emoji": "🧠",
    "color": WH,
    "tagline": "★ The flagship diagnostic — your most sellable service",
    "inputs": {
        "company_name": "Company name",
        "employees":    "Number of employees",
        "industry":     "Industry / vertical",
        "services":     "Products or services offered",
        "revenue":      "Annual revenue (optional)",
    },
    "system": """You are a world-class AI Business Consultant. A company has hired you for a premium diagnostic session.
Deliver an authoritative, detailed, and highly actionable consulting report.

Respond with this EXACT structure:

## BUSINESS OVERVIEW & AI READINESS SCORE
Score the company 1–10 across 5 dimensions: Data Maturity, Process Complexity,
Tech Stack, Team Capacity, Leadership Buy-in. Justify each score.

## TOP 10 OPERATIONAL PAIN POINTS
For each:
- Pain point name
- Current situation description
- Business cost (time + money estimate)
- AI solution that fixes it
- Build complexity: Low / Med / High

## AI AGENTS THAT WOULD TRANSFORM THIS BUSINESS
| Agent Name | Purpose | Est. Build Time | Monthly Value | Priority |
List 7–8 agents.

## ESTIMATED COST SAVINGS BREAKDOWN
| Area | Current Annual Cost | After AI | Annual Savings |
Total savings at bottom.

## IMPLEMENTATION ROADMAP
- Weeks 1–2: Foundation
- Month 1: First automations live
- Month 2–3: Core systems running
- Month 4–6: Optimization & scale
- Month 6–12: Full autonomous operation

## TECHNOLOGY STACK RECOMMENDATION
Exactly which tools, APIs, and platforms to build on and why.

## INVESTMENT & ROI SUMMARY
- Estimated implementation cost: $X–$X
- Annual savings: $X
- Revenue impact: $X
- First-year ROI: X%
- Payback period: X months

## IMMEDIATE ACTION ITEMS
Top 5 things to do in the next 30 days. Numbered and specific.

This is a premium paid deliverable. Be authoritative, specific, and thorough.""",
}

# ── DISPLAY HELPERS ───────────────────────────────────────────────────────────
def banner():
    print(f"""
{YL}{'═' * 64}{R}
{BD}  AI CONSULTING AGENT STACK{R}  {DM}v1.0 · Powered by Claude{R}
{YL}{'═' * 64}{R}
{DM}  7 specialist agents + 1 flagship diagnostic
  One-person agency · $25k–$80k/mo potential
  Outputs saved to: {OUTPUT_DIR}/{R}
{YL}{'─' * 64}{R}
""")

def agent_menu():
    all_agents = AGENTS + [FLAGSHIP_AGENT]
    print(f"\n{BD}  SELECT AN AGENT:{R}\n")
    for i, a in enumerate(all_agents, 1):
        flag = f"  {YL}★ FLAGSHIP{R}" if a["id"] == "ai_consultant" else ""
        print(f"  {a['color']}{BD}{i:2}. {a['emoji']}  {a['name']}{R}{flag}")
        print(f"      {DM}{a['tagline']}{R}")
    print(f"\n  {DM} 0. Run full pipeline (all 7 core agents){R}\n")
    return all_agents

def prompt_inputs(agent: dict) -> dict:
    print(f"\n{agent['color']}{BD}  {agent['emoji']} {agent['name']} — Inputs{R}")
    print(f"  {YL}{'─' * 50}{R}\n")
    values = {}
    for key, label in agent["inputs"].items():
        is_optional = "optional" in label.lower()
        suffix = f" {DM}(Enter to skip){R}" if is_optional else f" {YL}*{R}"
        val = input(f"  {label}{suffix}: ").strip()
        values[key] = val if val else "Not provided"
    return values

def spinner(msg: str):
    stop = threading.Event()
    frames = ["⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏"]
    def _spin():
        i = 0
        while not stop.is_set():
            print(f"\r  {YL}{frames[i % len(frames)]}{R} {DM}{msg}{R}", end="", flush=True)
            time.sleep(0.1)
            i += 1
    t = threading.Thread(target=_spin, daemon=True)
    t.start()
    return stop, t

def stop_spinner(stop, thread):
    stop.set()
    thread.join()
    print("\r" + " " * 70 + "\r", end="", flush=True)

def preview(text: str, n: int = 10):
    lines = text.strip().split("\n")
    print()
    for line in lines[:n]:
        if line.startswith("## "):
            print(f"  {YL}{BD}{line[3:]}{R}")
        elif line.startswith("### "):
            print(f"  {BD}{line[4:]}{R}")
        elif line.strip():
            wrapped = textwrap.fill(line, width=68, initial_indent="  ", subsequent_indent="  ")
            print(f"{DM}{wrapped}{R}")
    if len(lines) > n:
        print(f"  {DM}... [{len(lines) - n} more lines — see saved file]{R}")
    print()

# ── CORE RUNNER ───────────────────────────────────────────────────────────────
def run_agent(agent: dict, inputs: dict, context: str = "") -> str:
    input_block = "\n".join(f"- {k.replace('_', ' ').title()}: {v}" for k, v in inputs.items())
    user_msg = f"""CLIENT / COMPANY INPUTS:
{input_block}

{"CONTEXT FROM PREVIOUS AGENTS:\n" + context if context else ""}

Deliver your complete professional output now."""

    resp = client.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        system=agent["system"],
        messages=[{"role": "user", "content": user_msg}],
    )
    return resp.content[0].text

# ── SAVE OUTPUTS ──────────────────────────────────────────────────────────────
def save(agent: dict, inputs: dict, output: str) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    fname = OUTPUT_DIR / f"{ts}_{agent['id']}.md"
    header = (
        f"# {agent['emoji']} {agent['name']}\n\n"
        f"*Generated: {datetime.now().strftime('%B %d, %Y %H:%M')}*\n\n"
        f"## INPUTS\n"
        + "\n".join(f"- **{k}**: {v}" for k, v in inputs.items())
        + "\n\n---\n\n"
    )
    fname.write_text(header + output + "\n", encoding="utf-8")
    return fname

def save_pipeline(results: list) -> Path:
    OUTPUT_DIR.mkdir(exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    master = OUTPUT_DIR / f"{ts}_FULL_PIPELINE.md"
    parts = [f"# 🗂 FULL CONSULTING PIPELINE REPORT\n\n*{datetime.now().strftime('%B %d, %Y %H:%M')}*\n\n---\n"]
    for agent, inputs, output in results:
        parts.append(f"\n\n# {agent['emoji']} {agent['name'].upper()}\n\n{output}\n\n---")
    master.write_text("".join(parts), encoding="utf-8")
    return master

# ── SINGLE AGENT MODE ─────────────────────────────────────────────────────────
def run_single(agent: dict):
    print(f"\n{agent['color']}{BD}  {'═' * 52}{R}")
    print(f"  {agent['emoji']}  {BD}{agent['name']}{R}")
    print(f"  {DM}{agent['tagline']}{R}")
    print(f"{agent['color']}{BD}  {'═' * 52}{R}")

    inputs = prompt_inputs(agent)

    s, t = spinner(f"Running {agent['name']}...")
    try:
        output = run_agent(agent, inputs)
        stop_spinner(s, t)
        print(f"  {GR}{BD}✓ Complete{R}  {DM}({len(output.split())} words){R}")
        preview(output)
        fpath = save(agent, inputs, output)
        print(f"  {YL}💾 Saved:{R} {BD}{fpath}{R}\n")
    except Exception as e:
        stop_spinner(s, t)
        print(f"  {RD}✗ Error: {e}{R}\n")

# ── FULL PIPELINE MODE ────────────────────────────────────────────────────────
def run_pipeline():
    print(f"\n{YL}{BD}  🚀 FULL PIPELINE — All 7 Core Agents{R}")
    print(f"  {DM}Each agent receives a summary of previous agents' outputs.{R}\n")

    print(f"  {BD}Enter your client/company brief:{R}\n")
    brief = {
        "company_name":  input("  Company name: ").strip() or "Unknown Company",
        "industry":      input("  Industry: ").strip() or "General",
        "employees":     input("  Employees: ").strip() or "Unknown",
        "services":      input("  Services/products: ").strip() or "Unknown",
        "target_market": input("  Target market: ").strip() or "Unknown",
        "geography":     input("  Geography: ").strip() or "US",
    }

    results = []
    context = ""

    for i, agent in enumerate(AGENTS, 1):
        print(f"\n{agent['color']}{BD}  [{i}/{len(AGENTS)}] {agent['emoji']} {agent['name']}{R}")
        print(f"  {DM}{agent['tagline']}{R}")
        s, t = spinner(f"Generating output...")
        try:
            output = run_agent(agent, brief, context)
            stop_spinner(s, t)
            print(f"  {GR}✓ Done{R}  {DM}({len(output.split())} words){R}")
            preview(output, n=5)
            results.append((agent, brief, output))
            context += f"\n[{agent['name']}]: {output[:400]}...\n"
        except Exception as e:
            stop_spinner(s, t)
            print(f"  {RD}✗ Error: {e}{R}")

    master = save_pipeline(results)
    print(f"\n{YL}{'═' * 64}{R}")
    print(f"{BD}  ✦ Pipeline complete! {len(results)}/{len(AGENTS)} agents ran.{R}")
    print(f"  {YL}📖 Master report:{R} {BD}{master}{R}")
    print(f"{YL}{'═' * 64}{R}\n")

# ── INTERACTIVE MENU ──────────────────────────────────────────────────────────
def interactive():
    banner()
    all_agents = agent_menu()

    try:
        choice = input(f"  {BD}Select (0–{len(all_agents)}): {R}").strip()
    except (KeyboardInterrupt, EOFError):
        print(f"\n  {DM}Goodbye.{R}\n")
        sys.exit(0)

    if choice == "0":
        run_pipeline()
    else:
        try:
            idx = int(choice) - 1
            if 0 <= idx < len(all_agents):
                run_single(all_agents[idx])
            else:
                print(f"  {RD}Invalid choice.{R}\n")
        except ValueError:
            print(f"  {RD}Invalid input.{R}\n")

# ── CLI MODE ──────────────────────────────────────────────────────────────────
def cli():
    parser = argparse.ArgumentParser(
        description="AI Consulting Agent Stack — 7 agents + flagship diagnostic",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
AGENT IDs:
  lead_gen | outreach | discovery | proposal |
  market_intel | automation_builder | client_reporting | ai_consultant | all

EXAMPLES:
  python consulting_agent_stack.py

  python consulting_agent_stack.py --agent discovery \\
      --company "Acme Corp" --industry "Retail" \\
      --employees "75" --services "E-commerce fulfillment"

  python consulting_agent_stack.py --agent ai_consultant \\
      --company "GreenLeaf" --industry "Logistics" \\
      --employees "120" --services "Last-mile delivery"

  python consulting_agent_stack.py --agent all
        """
    )
    parser.add_argument("--agent",         help="Agent ID or 'all'")
    parser.add_argument("--company",       default="Client Company")
    parser.add_argument("--industry",      default="General Business")
    parser.add_argument("--employees",     default="50")
    parser.add_argument("--services",      default="Various services")
    parser.add_argument("--revenue",       default="Not provided")
    parser.add_argument("--pain_points",   default="To be analyzed")
    parser.add_argument("--budget",        default="$5,000–$20,000")
    parser.add_argument("--timeline",      default="90 days")
    parser.add_argument("--geography",     default="US")
    parser.add_argument("--service_type",  default="AI automation consulting")
    parser.add_argument("--prospect_role", default="COO")
    parser.add_argument("--pain_point",    default="Operational inefficiency")
    parser.add_argument("--your_service",  default="AI workflow automation")
    parser.add_argument("--company_stage", default="growing")
    parser.add_argument("--process_name",  default="Manual reporting")
    parser.add_argument("--process_desc",  default="Staff manually compile reports")
    parser.add_argument("--inputs_outputs",default="Raw data in, formatted report out")
    parser.add_argument("--tools_used",    default="Excel, Email")
    parser.add_argument("--project_type",  default="Workflow automation")
    parser.add_argument("--metrics",       default="Hours saved, cost reduction")
    parser.add_argument("--period",        default="Month 1")
    parser.add_argument("--results_data",  default="Early results pending")

    args = parser.parse_args()

    if not args.agent:
        interactive()
        return

    if args.agent == "all":
        run_pipeline()
        return

    all_agents = AGENTS + [FLAGSHIP_AGENT]
    agent = next((a for a in all_agents if a["id"] == args.agent), None)
    if not agent:
        ids = " | ".join(a["id"] for a in all_agents)
        print(f"{RD}Unknown agent '{args.agent}'.{R}\nValid IDs: {ids}")
        sys.exit(1)

    # Map CLI args → agent inputs
    arg_map = vars(args)
    inputs = {k: arg_map.get(k, "Not provided") for k in agent["inputs"]}

    banner()
    print(f"{agent['color']}{BD}  {agent['emoji']} {agent['name']}{R}  {DM}{agent['tagline']}{R}\n")

    s, t = spinner(f"Running {agent['name']}...")
    try:
        output = run_agent(agent, inputs)
        stop_spinner(s, t)
        print(f"  {GR}{BD}✓ Complete{R}  {DM}({len(output.split())} words){R}")
        preview(output)
        fpath = save(agent, inputs, output)
        print(f"  {YL}💾 Saved:{R} {BD}{fpath}{R}\n")
    except Exception as e:
        stop_spinner(s, t)
        print(f"  {RD}✗ Error: {e}{R}\n")
        sys.exit(1)

# ── ENTRY POINT ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    cli()
