"""
================================================================================
data/generate_hr_documents.py  —  Synthetic HR Policy Document Generator
================================================================================

Purpose:
    Generates realistic synthetic HR policy documents for testing the RAG
    knowledge base without requiring real company documents.

    Creates:
        • employee_handbook.txt        — comprehensive employee handbook
        • pto_leave_policy.txt         — PTO, sick, parental leave policy
        • benefits_guide.txt           — health, 401(k), and benefits overview
        • code_of_conduct.txt          — conduct, ethics, and compliance
        • performance_review_guide.txt — performance management process
        • onboarding_guide.txt         — new hire onboarding procedures
        • compensation_policy.txt      — pay, bonuses, and equity
================================================================================
"""

import os
from pathlib import Path

DOCUMENTS_DIR = "knowledge_base/documents"


def write_doc(filename: str, content: str) -> None:
    path = Path(DOCUMENTS_DIR) / filename
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    print(f"  Generated: {filename}")


def generate_all() -> None:
    print(f"Generating HR documents in '{DOCUMENTS_DIR}/'...")

    # ── PTO & Leave Policy ────────────────────────────────────────────────────
    write_doc("pto_leave_policy.txt", """
ACME CORPORATION — PAID TIME OFF AND LEAVE POLICY
Last Updated: January 2026

1. PAID TIME OFF (PTO)

1.1 Accrual Rate
Full-time employees accrue 1.75 days of PTO per month (21 days per year).
Part-time employees (minimum 20 hours/week) accrue PTO on a pro-rated basis
based on their FTE percentage.

New employees begin accruing PTO from their first day of employment.
PTO may be used after completing 90 days of employment.

1.2 Carryover
Employees may carry over a maximum of 5 PTO days into the following calendar year.
Unused PTO above the 5-day carryover cap is forfeited on January 1 of each year.
Employees will receive a reminder email in November if they are at risk of forfeiture.

1.3 PTO Request Process
Employees must submit PTO requests through Workday at least:
  - 1 business day in advance for absences of 1-2 days
  - 2 weeks in advance for absences of 3-5 days
  - 4 weeks in advance for absences exceeding 5 days

Manager approval is required for all PTO requests. Requests are approved on a
first-come, first-served basis, subject to business needs.

1.4 PTO Payout on Termination
Upon separation from the company, employees will be paid out for all accrued,
unused PTO in their final paycheck. This applies to both voluntary and
involuntary separations.

2. SICK LEAVE

2.1 Separate Sick Leave Bank
Acme provides a separate sick leave bank of 10 days per year (not accrued,
available on January 1). Sick leave does not carry over and is not paid out
upon termination.

2.2 Approved Uses of Sick Leave
  - Employee's own illness, injury, or medical appointment
  - Care for an immediate family member (spouse, child, parent, domestic partner)
  - Mental health days (2 per year without medical documentation required)
  - Preventive care appointments

2.3 Documentation Requirements
For absences exceeding 3 consecutive business days, employees must provide a
physician's note confirming the need for leave. Documentation must be submitted
to HR within 5 business days of returning to work.

3. PARENTAL LEAVE

3.1 Primary Caregiver Leave
Full-time employees who are primary caregivers of a new child (birth, adoption,
or foster placement) are entitled to 16 weeks of paid parental leave at 100%
of base salary.

3.2 Secondary Caregiver Leave
Secondary caregivers are entitled to 6 weeks of paid parental leave at 100%
of base salary.

3.3 Eligibility
Employees must have completed 12 months of continuous employment to qualify for
full parental leave pay. Employees with 6-12 months of tenure receive 8 weeks
paid leave. Employees with less than 6 months receive unpaid FMLA leave.

3.4 Benefits During Leave
All employer-paid benefits (health insurance, life insurance) continue during
parental leave. Employees are responsible for their portion of health insurance
premiums during leave.

4. BEREAVEMENT LEAVE

Employees are entitled to:
  - 5 business days for death of immediate family (spouse, child, parent, sibling)
  - 3 business days for death of extended family (grandparent, in-law, aunt/uncle)
  - 1 business day for death of close friend or distant relative

Additional unpaid leave may be approved by the manager for exceptional circumstances.

5. JURY DUTY AND MILITARY LEAVE

Acme provides full pay for up to 15 days of jury duty per year.
Military leave is provided in accordance with the Uniformed Services Employment
and Reemployment Rights Act (USERRA).

6. UNPAID LEAVE

Employees may request unpaid personal leave of up to 12 weeks with manager
and HR approval. Benefits continue during approved unpaid leave if the employee
pays both the employee and employer portion of premiums.

For questions about leave policies, contact HR at hr@acme.com or ext. 5000.
""")

    # ── Benefits Guide ────────────────────────────────────────────────────────
    write_doc("benefits_guide.txt", """
ACME CORPORATION — EMPLOYEE BENEFITS GUIDE 2026
Effective: January 1, 2026

HEALTH INSURANCE

We offer three health plan options through BlueCross BlueShield:

1. PPO Plan (Preferred Provider Organisation)
   - Monthly Premium (Employee Only):   $120/month
   - Monthly Premium (Employee + 1):    $280/month
   - Monthly Premium (Family):          $400/month
   - Deductible: $1,000 / $2,000 / $3,000
   - Out-of-Pocket Maximum: $4,000 / $8,000 / $12,000
   - In-Network Copay: $25 per visit
   - Prescription: $10 generic / $40 brand

2. HDHP + HSA (High-Deductible Health Plan)
   - Monthly Premium (Employee Only):   $60/month
   - Monthly Premium (Employee + 1):    $140/month
   - Monthly Premium (Family):          $200/month
   - Deductible: $2,800 individual / $5,600 family
   - Out-of-Pocket Maximum: $5,000 / $10,000 / $14,000
   - HSA-eligible: Yes. Acme contributes $500/year to your HSA.
   - All services apply to deductible first; no copays.

3. HMO Plan (Health Maintenance Organisation)
   - Monthly Premium (Employee Only):   $80/month
   - Monthly Premium (Employee + 1):    $200/month
   - Monthly Premium (Family):          $320/month
   - Deductible: $500 / $1,000 / $1,500
   - Out-of-Pocket Maximum: $3,000 / $6,000 / $9,000
   - In-Network Copay: $20 per visit
   - Requires primary care physician selection.

DENTAL AND VISION

Dental: Provided through Delta Dental.
  - Monthly premium: $15 (employee), $35 (+1), $55 (family)
  - 2 cleanings/year covered 100%
  - Basic restorative: 80% after deductible
  - Major restorative/orthodontia: 50% with $2,000 annual maximum

Vision: Provided through VSP.
  - Monthly premium: $8 (employee), $16 (+1), $22 (family)
  - Annual eye exam: $10 copay
  - Frames allowance: $150/year
  - Contact lens allowance: $150/year

RETIREMENT — 401(K) PLAN

Provider: Fidelity Investments
IRS Contribution Limit (2026): $23,500 (under 50) / $31,000 (50+)

Employer Match:
  - 100% match on the first 3% of salary you contribute
  - 50% match on the next 2% of salary you contribute
  - Maximum employer match: 4% of salary

Example: If you earn $80,000 and contribute 5%, Acme contributes:
  - 100% of 3% = $2,400
  - 50% of 2% = $800
  - Total Acme contribution: $3,200/year

Vesting: 4-year graded vesting
  - Year 1: 25% vested
  - Year 2: 50% vested
  - Year 3: 75% vested
  - Year 4: 100% vested

To enrol: Log in to Fidelity NetBenefits at fidelity.com/acme

LIFE INSURANCE AND DISABILITY

Life Insurance:
  - Basic: 2x annual salary (fully employer-paid)
  - Supplemental: up to 5x salary (employee-paid, evidence of insurability required)
  - Dependent life: $25,000 for spouse, $10,000 per child (employee-paid)

Short-Term Disability:
  - 60% of salary, maximum $3,000/week
  - 14-day elimination period
  - Maximum benefit period: 12 weeks

Long-Term Disability:
  - 60% of salary, maximum $10,000/month
  - 90-day elimination period
  - Benefit period: To age 65

ADDITIONAL BENEFITS

Employee Assistance Programme (EAP):
  - 6 free confidential counselling sessions per year
  - 24/7 crisis support line: 1-800-EAP-ACME
  - Financial counselling, legal consultation (1 session each, no charge)

Wellness Benefits:
  - $50/month gym reimbursement (submit receipts through Expensify)
  - $280/month pre-tax commuter benefit (transit or parking)
  - On-site fitness centre (headquarters only)

Professional Development:
  - $2,000/year learning & development budget
  - Submit requests through Learning Management System (LMS)
  - Eligible: conferences, courses, certifications, books

Remote Work Stipend:
  - $500 annual home office equipment stipend
  - Reimbursable through Expensify under "WFH Equipment"

Open Enrolment: Benefits elections must be made annually during Open Enrolment
(November 1-15). New hires have 30 days from start date to enrol.
Outside of these windows, changes are only permitted after qualifying life events.
""")

    # ── Code of Conduct ───────────────────────────────────────────────────────
    write_doc("code_of_conduct.txt", """
ACME CORPORATION — CODE OF CONDUCT AND ETHICS POLICY
Effective: January 1, 2026

1. OUR VALUES

Acme is committed to conducting business with the highest ethical standards.
All employees, contractors, and vendors are expected to:
  - Act with integrity and honesty in all interactions
  - Treat all people with dignity and respect
  - Protect company and customer confidential information
  - Comply with all applicable laws and regulations
  - Report concerns without fear of retaliation

2. EQUAL OPPORTUNITY AND NON-DISCRIMINATION

Acme is an equal opportunity employer. We do not discriminate on the basis of:
race, colour, religion, national origin, sex, gender identity, sexual orientation,
age, disability, pregnancy, marital status, veteran status, or any other
characteristic protected by applicable law.

3. HARASSMENT AND HOSTILE WORK ENVIRONMENT

Acme has a zero-tolerance policy for harassment of any kind, including:
  - Sexual harassment (unwanted advances, comments, or contact)
  - Bullying or intimidation
  - Discriminatory jokes or comments
  - Retaliation against anyone who reports a concern

Any employee who experiences or witnesses harassment should report it to:
  - Their HR Business Partner directly
  - The anonymous Ethics Hotline: 1-800-ETHICS-1
  - The online reporting portal: ethics.acme.com

All reports are investigated promptly and confidentially.

4. CONFLICTS OF INTEREST

Employees must disclose any actual or potential conflict of interest to their
manager and HR. A conflict exists when personal interests could influence, or
appear to influence, business decisions. Examples include:
  - Financial interest in a vendor or competitor
  - Outside employment with a competitor
  - Romantic relationship with a direct report

Undisclosed conflicts of interest may result in disciplinary action up to and
including termination.

5. CONFIDENTIALITY AND DATA PROTECTION

Employees have access to confidential information including trade secrets,
customer data, employee records, and financial information. You must:
  - Not share confidential information outside the company without authorisation
  - Not use confidential information for personal gain
  - Comply with GDPR, CCPA, and other applicable privacy laws
  - Report suspected data breaches immediately to security@acme.com

6. SOCIAL MEDIA POLICY

Employees may maintain personal social media accounts but must:
  - Not share confidential company information
  - Not make statements that could be construed as official company positions
  - Not disparage customers, partners, or colleagues
  - Identify themselves as expressing personal views when discussing the company

7. DISCIPLINARY PROCESS

Policy violations are addressed through a progressive discipline process:
  1. Verbal counselling (documented)
  2. Written warning
  3. Performance Improvement Plan (PIP)
  4. Final written warning or suspension
  5. Termination

Serious violations (harassment, theft, fraud, safety violations) may result in
immediate termination without progressive steps.

8. REPORTING CONCERNS

The company encourages reporting of ethical concerns. You may report:
  - To your manager or HR Business Partner
  - To the Ethics Hotline (anonymous): 1-800-ETHICS-1
  - Via the ethics portal: ethics.acme.com

Retaliation against good-faith reporters is prohibited and itself a violation
of this Code subject to disciplinary action.
""")

    # ── Performance Review Guide ──────────────────────────────────────────────
    write_doc("performance_review_guide.txt", """
ACME CORPORATION — PERFORMANCE MANAGEMENT GUIDE
Effective: January 2026

1. PERFORMANCE REVIEW CYCLE

Annual Performance Reviews:
  - Review period: January 1 – December 31
  - Self-assessment due: January 15
  - Manager review due: January 31
  - Calibration sessions: February
  - Results communicated to employees: March 1
  - Merit increases effective: April 1

Mid-Year Check-In:
  - Informal check-in between employee and manager in July
  - Not a formal review; focused on goal progress and development

New Hire Reviews:
  - 30-day check-in (informal, with manager)
  - 90-day review (formal, determines whether employment continues)
  - First annual review at the standard cycle after 6+ months of tenure

2. PERFORMANCE RATINGS

Performance is rated on a 4-point scale:
  4 - Exceptional: Significantly exceeded all goals; role model performance
  3 - Exceeds Expectations: Consistently exceeded most goals
  2 - Meets Expectations: Met all goals; solid, reliable performance
  1 - Below Expectations: Did not meet key goals; improvement required

Calibration ensures ratings are consistent across teams and that the distribution
does not result in grade inflation. Target distribution: 10% Exceptional,
25% Exceeds, 55% Meets, 10% Below.

3. GOALS AND OKRs

Employees set 3-5 goals each year using the OKR (Objectives and Key Results) framework.
Goals should be SMART: Specific, Measurable, Achievable, Relevant, Time-bound.

Goals are entered in Workday by March 15 of each year and reviewed at mid-year.
Goal changes require manager approval.

4. MERIT INCREASES

Merit increases are based on performance rating and position in pay band:
  - Exceptional (4):        4-6% merit increase
  - Exceeds Expectations (3): 2-4% merit increase
  - Meets Expectations (2):   0-2% merit increase
  - Below Expectations (1):   0% merit increase; PIP required

Actual merit increases also depend on company financial performance.
The compensation team confirms the merit budget by February each year.

5. PERFORMANCE IMPROVEMENT PLANS (PIPs)

A PIP is initiated when an employee receives a Below Expectations rating or when
performance concerns arise outside of the review cycle. PIPs include:
  - Specific, measurable improvement goals
  - A clear timeline (typically 30-90 days)
  - Regular check-ins (at least weekly)
  - Resources and support to improve

Successful completion of a PIP results in returning to normal employment status.
Failure to meet PIP goals may result in termination.

6. PROMOTIONS

Promotions are typically aligned with the annual review cycle. To be considered:
  - Perform at the level of the next position for at least 6 months
  - Receive an Exceeds or Exceptional rating
  - Have a business need for the higher-level role
  - Obtain manager sponsorship

Off-cycle promotions can be approved by the VP of HR for exceptional circumstances.
""")

    # ── Onboarding Guide ──────────────────────────────────────────────────────
    write_doc("onboarding_guide.txt", """
ACME CORPORATION — NEW HIRE ONBOARDING GUIDE
Welcome to Acme!

BEFORE YOUR FIRST DAY

Pre-Arrival Checklist:
  - Sign your offer letter and employment agreement (Workday)
  - Complete I-9 identity verification (HR will send instructions)
  - Complete background check (via Checkr)
  - Review and acknowledge company policies in the onboarding portal
  - Your laptop and equipment will be shipped to your home address or ready at office

WEEK 1 PRIORITIES

Day 1:
  - Arrive at 9:00 AM (or join the virtual orientation Zoom link)
  - Collect your ID badge from reception (bring government-issued photo ID)
  - Attend the All-Company Orientation (9:30 AM – 1:00 PM)
  - Meet your manager and immediate team
  - Set up your laptop (IT will assist)
  - Attend your team welcome lunch

Days 2-5:
  - Complete mandatory training in the LMS (Code of Conduct, Data Privacy — both due by end of week 1)
  - Configure corporate email, Slack, and other tools
  - Set up direct deposit in Workday (required for first paycheck)
  - Enrol in benefits — you have 30 days; don't wait! (Workday > Benefits)
  - Meet with your onboarding buddy (assigned by HR)
  - Begin reviewing department documentation and wikis

FIRST 30 DAYS

Key Activities:
  - Complete all required compliance training
  - Attend all mandatory HR onboarding sessions
  - Schedule 1:1s with key stakeholders in your department
  - Shadow your teammates on their core workflows
  - Set your 90-day goals with your manager (enter in Workday by day 14)
  - Enrol in 401(k) through Fidelity NetBenefits (optional but recommended)
  - Complete your 30-day check-in with your HR Business Partner

30-Day Check-In Topics:
  - How is onboarding going? Any blockers?
  - Are you clear on your role and expectations?
  - Do you have all the tools and access you need?
  - Any questions about benefits or policies?

FIRST 90 DAYS

Your First 90-Day Plan should include:
  - Learning the core business (customers, products, strategy)
  - Understanding how your team operates
  - Completing your role-specific onboarding tasks
  - Delivering an early win or project contribution
  - Building relationships across the organisation

At Day 90, you will have a formal check-in with your manager. This conversation
determines whether your employment continues and sets the stage for your first full year.

SYSTEMS AND TOOLS ACCESS

Core Systems (provisioned by IT on Day 1):
  - Workday (HRIS — time off, benefits, payroll)
  - Slack (internal communication)
  - Google Workspace or Microsoft 365 (email, calendar, docs)
  - Zoom (video conferencing)
  - Jira/Asana (project management — department-specific)
  - Confluence (company wiki)

Role-Specific Systems (provisioned by IT within 48 hours):
  - Engineers: GitHub, AWS/GCP/Azure, Datadog, development tools
  - Sales: Salesforce, Outreach, LinkedIn Sales Navigator
  - Finance: NetSuite, Expensify, Coupa
  - Marketing: HubSpot, Figma, Marketo

If you are missing access to any system, submit an IT ticket at it-help.acme.com.

PAYROLL AND EXPENSES

Payroll: Bi-weekly, paid on Fridays. First paycheck may be delayed by one pay cycle.
Expense Reimbursement: Submit via Expensify within 30 days of incurring the expense.
Manager approval required for all expenses over $50.

CONTACTS

Your HR Business Partner: hr-bp@acme.com | ext. 5001
IT Helpdesk: it-help@acme.com | it-help.acme.com | ext. 5100
Office Manager: facilities@acme.com | ext. 5200
Benefits Questions: benefits@acme.com | ext. 5002
Payroll: payroll@acme.com | ext. 5003

Welcome aboard! We're glad you're here.
""")

    # ── Compensation Policy ───────────────────────────────────────────────────
    write_doc("compensation_policy.txt", """
ACME CORPORATION — COMPENSATION AND PAY POLICY
Effective: January 1, 2026

1. PAY PHILOSOPHY

Acme targets the 60th percentile of the relevant market for base salary.
We conduct annual compensation benchmarking using Radford and Mercer surveys.
Total compensation includes base salary, performance bonus, and equity awards.

2. BASE SALARY

Pay Bands:
Acme uses career levels from L1 (entry-level) to L7 (principal/distinguished).
Each level has a defined pay band with a minimum, midpoint, and maximum.
Employees are expected to reach the midpoint of their band within 2-3 years
of being in the level.

Pay Review:
Annual merit increases are effective April 1 based on performance ratings.
See the Performance Management Guide for merit increase guidelines.
Off-cycle adjustments can be approved by the Compensation team for
retention, equity, or market correction purposes.

3. PERFORMANCE BONUS

Annual Bonus Programme:
  - Eligible employees: all full-time employees who have been employed for
    at least 6 months by December 31
  - Target bonus as % of base salary:
      L1-L3 (individual contributors): 10%
      L4-L5 (senior individual contributors): 15%
      L6+ (principal, staff, managers): 20%
      Directors: 25%
      VPs and above: 30-50% (as per individual agreements)

  - Actual bonus = Target Bonus × Company Performance Multiplier × Individual Rating Multiplier
  - Company Performance Multiplier: 0-1.5× based on achievement of company OKRs
  - Individual Multiplier: 0.8× (Meets) to 1.25× (Exceptional)
  - Bonuses are paid in March for the prior year's performance

4. EQUITY COMPENSATION

Equity Awards:
New hire equity grants vest over 4 years with a 1-year cliff:
  - 25% vests on the 1-year anniversary of grant date
  - 1/36 vests monthly for the following 36 months

Refresh Grants:
Annual refresh grants may be awarded based on performance rating:
  - Exceptional: 150% of target refresh
  - Exceeds: 100% of target refresh
  - Meets: 50% of target refresh
  - Below: 0%

Exercise and Tax:
  - Equity information is managed through Carta
  - Consult a financial advisor for tax implications of option exercises
  - Exercise deadline after termination: 90 days (ISO options)

5. EXPENSE REIMBURSEMENT

Reimbursable expenses include:
  - Business travel (economy class for flights under 5 hours)
  - Client entertainment (pre-approval required over $100)
  - Home office equipment ($500/year stipend)
  - Professional development ($2,000/year)
  - Phone bill: $50/month for employees who use personal phone for business

Submission: via Expensify within 30 days of incurrence.
Out-of-policy expenses will not be reimbursed.

6. PAYROLL

Pay Schedule: Bi-weekly (26 pay periods per year), paid on Fridays.
Direct Deposit: Required. Set up in Workday before your first paycheck.
Pay Stubs: Available in Workday under "Pay" > "Pay History."
W-2 Forms: Issued by January 31 each year.

For compensation questions: compensation@acme.com
For payroll questions: payroll@acme.com | ext. 5003
""")

    print(f"\nDone — generated {len(os.listdir(DOCUMENTS_DIR))} HR documents.")


if __name__ == "__main__":
    generate_all()
