"""
================================================================================
agents/hr_policy_agent.py  —  HR Policy Question-Answering Agent
================================================================================

Purpose:
    Conversational HR assistant that answers employee policy questions with
    source citations and escalation routing.  Built on top of the
    KnowledgeBaseAgent RAG engine but adds:

        • Intent classification  — detects question category (leave, benefits,
                                   conduct, compensation, etc.)
        • Conversation memory    — maintains context across multi-turn exchanges
        • Confidence-based routing — low-confidence answers escalate to HR team
        • Policy change detection — flags when a question references outdated info
        • Sensitive query handling — routes to HR partner for disciplinary, legal,
                                     or medical questions

    This is the layer that employees interact with, either via the REST API,
    Teams/Slack bot, or the web portal.  It handles the full conversation
    lifecycle from greeting to escalation.

Intent categories:
    leave         — PTO, sick leave, parental leave, FMLA, sabbatical
    benefits      — health insurance, dental, vision, 401(k), HSA, EAP
    compensation  — salary, bonuses, equity, pay cycles, expense reimbursement
    conduct       — code of conduct, harassment, disciplinary process
    performance   — reviews, PIPs, promotion, feedback processes
    onboarding    — new hire process, equipment, access, orientation
    offboarding   — resignation, notice period, final pay, COBRA
    general       — anything that does not match a specific category

Usage:
    agent    = HRPolicyAgent()
    response = agent.answer(
        question     = "How many sick days do I get per year?",
        employee_id  = "emp-123",
        conversation_id = "conv-456",
    )
    print(response.answer)
    print(response.should_escalate)
================================================================================
"""

from __future__ import annotations

import logging
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional

from agents.knowledge_base_agent import KnowledgeBaseAgent, RAGResult

logger = logging.getLogger("platform.hr_policy")

# ── Confidence threshold below which we escalate to HR team ──────────────────
ESCALATION_THRESHOLD = 0.40

# ── Sensitive topics that always route to a human HR partner ─────────────────
ALWAYS_ESCALATE_PATTERNS = [
    r"\b(harassment|discrimination|hostile|assault|abuse|retaliation)\b",
    r"\b(disciplinary|termination|fired|wrongful|lawsuit|attorney|lawyer)\b",
    r"\b(mental health|depression|anxiety|disability|accommodation|ada)\b",
    r"\b(salary negotiation|counter offer|pay raise request)\b",
    r"\b(whistleblow|report manager|report supervisor)\b",
]

# ── Intent → keyword mapping for classification ───────────────────────────────
INTENT_KEYWORDS: Dict[str, List[str]] = {
    "leave":        ["pto", "vacation", "sick", "leave", "fmla", "parental", "maternity",
                     "paternity", "sabbatical", "time off", "absence", "holiday"],
    "benefits":     ["benefits", "insurance", "health", "dental", "vision", "401k",
                     "hsa", "fsa", "eap", "retirement", "pension", "gym", "wellness"],
    "compensation": ["salary", "pay", "bonus", "raise", "equity", "stock", "rsu",
                     "expense", "reimbursement", "paycheck", "payroll", "commission"],
    "conduct":      ["conduct", "policy", "code", "ethics", "harassment", "bullying",
                     "complaint", "grievance", "discipline", "termination"],
    "performance":  ["review", "performance", "feedback", "pip", "promotion",
                     "raise", "evaluation", "goals", "okr"],
    "onboarding":   ["onboard", "new hire", "first day", "start", "orientation",
                     "equipment", "laptop", "badge", "access", "setup"],
    "offboarding":  ["resign", "quit", "notice", "last day", "offboard",
                     "cobra", "final pay", "severance", "transition"],
}


@dataclass
class PolicyResponse:
    """
    Full response from the HR Policy Agent.

    Attributes:
        question:          The employee's original question.
        answer:            The policy answer with citations.
        intent:            Detected intent category.
        citations:         Source document references.
        confidence:        Retrieval confidence score (0-1).
        should_escalate:   True if the question should be routed to HR team.
        escalation_reason: Why escalation was triggered (if applicable).
        conversation_id:   Multi-turn conversation identifier.
        employee_id:       Employee who asked the question.
        timestamp:         ISO UTC timestamp.
        response_id:       Unique response identifier.
        follow_up_prompts: Suggested follow-up questions for the employee.
    """
    question:          str
    answer:            str
    intent:            str
    citations:         List[Dict[str, Any]]
    confidence:        float
    should_escalate:   bool
    escalation_reason: str
    conversation_id:   str
    employee_id:       str
    timestamp:         str
    response_id:       str
    follow_up_prompts: List[str] = field(default_factory=list)
    latency_ms:        float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        return {
            "response_id":       self.response_id,
            "conversation_id":   self.conversation_id,
            "employee_id":       self.employee_id,
            "question":          self.question,
            "answer":            self.answer,
            "intent":            self.intent,
            "citations":         self.citations,
            "confidence":        round(self.confidence, 3),
            "should_escalate":   self.should_escalate,
            "escalation_reason": self.escalation_reason,
            "follow_up_prompts": self.follow_up_prompts,
            "timestamp":         self.timestamp,
            "latency_ms":        round(self.latency_ms, 2),
        }


class ConversationMemory:
    """
    Maintains per-employee conversation history for multi-turn Q&A.

    Stores the last N turns per conversation so the LLM has context
    when an employee asks a follow-up like "what about part-time employees?"

    Attributes:
        max_turns:  Maximum number of turns to retain per conversation.
        _store:     Dict mapping conversation_id → list of turn dicts.
    """

    def __init__(self, max_turns: int = 6) -> None:
        self.max_turns = max_turns
        self._store: Dict[str, List[Dict]] = {}

    def add_turn(
        self,
        conversation_id: str,
        question:        str,
        answer:          str,
        intent:          str,
    ) -> None:
        """Add a Q&A turn to the conversation history."""
        history = self._store.setdefault(conversation_id, [])
        history.append({
            "question":  question,
            "answer":    answer[:500],   # truncate to save memory
            "intent":    intent,
            "timestamp": datetime.utcnow().isoformat(),
        })
        # Keep only the last max_turns
        self._store[conversation_id] = history[-self.max_turns:]

    def get_history(self, conversation_id: str) -> List[Dict]:
        """Return the conversation history for a given conversation ID."""
        return self._store.get(conversation_id, [])

    def clear(self, conversation_id: str) -> None:
        """Clear history for a conversation (e.g., after session ends)."""
        self._store.pop(conversation_id, None)


class HRPolicyAgent:
    """
    Conversational HR policy assistant with intent routing and escalation.

    Attributes:
        kb:           KnowledgeBaseAgent (RAG engine).
        memory:       ConversationMemory for multi-turn context.
        escalation_threshold: Confidence below which answers escalate to HR.
    """

    def __init__(
        self,
        knowledge_base:       Optional[KnowledgeBaseAgent] = None,
        escalation_threshold: float = ESCALATION_THRESHOLD,
    ) -> None:
        self.kb                   = knowledge_base or KnowledgeBaseAgent()
        self.memory               = ConversationMemory(max_turns=6)
        self.escalation_threshold = escalation_threshold

    # ─────────────────────────────────────────────────────────────────────────
    # Public API
    # ─────────────────────────────────────────────────────────────────────────

    def answer(
        self,
        question:        str,
        employee_id:     str = "anonymous",
        conversation_id: Optional[str] = None,
    ) -> PolicyResponse:
        """
        Answer an employee's HR policy question.

        Flow:
            1. Detect intent category.
            2. Check for sensitive patterns → immediate escalation.
            3. Retrieve and generate via RAG.
            4. Check confidence → escalate if below threshold.
            5. Generate follow-up prompts.
            6. Store in conversation memory.

        Args:
            question:        Employee's natural-language question.
            employee_id:     Employee identifier for logging and personalisation.
            conversation_id: Existing conversation ID for multi-turn context.

        Returns:
            PolicyResponse with answer, citations, and routing decision.
        """
        t0              = time.perf_counter()
        conv_id         = conversation_id or str(uuid.uuid4())
        response_id     = str(uuid.uuid4())[:8]

        # ── Step 1: Classify intent ───────────────────────────────────────────
        intent = self._classify_intent(question)

        # ── Step 2: Check for always-escalate patterns ────────────────────────
        sensitive, sensitive_reason = self._check_sensitive(question)
        if sensitive:
            return self._escalation_response(
                question        = question,
                reason          = sensitive_reason,
                intent          = intent,
                employee_id     = employee_id,
                conversation_id = conv_id,
                response_id     = response_id,
                t0              = t0,
            )

        # ── Step 3: Enrich question with conversation context ─────────────────
        enriched_question = self._enrich_with_context(question, conv_id)

        # ── Step 4: RAG retrieval and generation ──────────────────────────────
        rag_result = self.kb.query(enriched_question, doc_type=self._intent_to_doc_type(intent))

        # ── Step 5: Low-confidence escalation ────────────────────────────────
        if not rag_result.found_relevant or rag_result.confidence < self.escalation_threshold:
            return self._escalation_response(
                question        = question,
                reason          = (
                    f"No relevant policy found with sufficient confidence "
                    f"(score: {rag_result.confidence:.2f}). "
                    "Please contact your HR Business Partner directly."
                ),
                intent          = intent,
                employee_id     = employee_id,
                conversation_id = conv_id,
                response_id     = response_id,
                t0              = t0,
            )

        # ── Step 6: Generate follow-up prompts ────────────────────────────────
        follow_ups = self._generate_follow_ups(intent, question)

        # ── Step 7: Store in memory ───────────────────────────────────────────
        self.memory.add_turn(conv_id, question, rag_result.answer, intent)

        latency = (time.perf_counter() - t0) * 1000

        logger.info(
            "Policy answer  employee=%s  intent=%s  confidence=%.3f  latency=%.1fms",
            employee_id, intent, rag_result.confidence, latency,
        )

        return PolicyResponse(
            question          = question,
            answer            = rag_result.answer,
            intent            = intent,
            citations         = rag_result.citations,
            confidence        = rag_result.confidence,
            should_escalate   = False,
            escalation_reason = "",
            conversation_id   = conv_id,
            employee_id       = employee_id,
            timestamp         = datetime.utcnow().isoformat(),
            response_id       = response_id,
            follow_up_prompts = follow_ups,
            latency_ms        = latency,
        )

    # ─────────────────────────────────────────────────────────────────────────
    # Intent classification
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _classify_intent(question: str) -> str:
        """
        Rule-based intent classifier using keyword matching.

        Returns the intent with the most keyword matches.
        Falls back to 'general' if no category has any matches.
        """
        q_lower = question.lower()
        scores  = {}
        for intent, keywords in INTENT_KEYWORDS.items():
            scores[intent] = sum(1 for kw in keywords if kw in q_lower)
        best = max(scores, key=lambda k: scores[k])
        return best if scores[best] > 0 else "general"

    @staticmethod
    def _intent_to_doc_type(intent: str) -> Optional[str]:
        """Map intent to a doc_type filter for the vector store query."""
        # Most intents map to 'policy'; onboarding has its own doc type
        return "onboarding" if intent == "onboarding" else None

    # ─────────────────────────────────────────────────────────────────────────
    # Sensitive topic detection
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _check_sensitive(question: str) -> tuple[bool, str]:
        """Return (is_sensitive, reason) for always-escalate patterns."""
        q_lower = question.lower()
        for pattern in ALWAYS_ESCALATE_PATTERNS:
            match = re.search(pattern, q_lower, re.IGNORECASE)
            if match:
                topic = match.group(0).strip()
                return True, (
                    f"Your question mentions '{topic}', which requires a confidential "
                    "conversation with your HR Business Partner. This ensures your "
                    "privacy and that you receive the appropriate support."
                )
        return False, ""

    # ─────────────────────────────────────────────────────────────────────────
    # Conversation context enrichment
    # ─────────────────────────────────────────────────────────────────────────

    def _enrich_with_context(self, question: str, conversation_id: str) -> str:
        """
        Prepend recent conversation history to the query for multi-turn context.

        This is what makes follow-up questions work — "what about part-timers?"
        is enriched to "Regarding sick leave policy, what about part-time employees?"
        """
        history = self.memory.get_history(conversation_id)
        if not history:
            return question

        # Include only the most recent prior Q for context
        last_turn = history[-1]
        context   = f"Previous question: {last_turn['question']}\n\nFollow-up: {question}"
        return context

    # ─────────────────────────────────────────────────────────────────────────
    # Follow-up prompt generation
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _generate_follow_ups(intent: str, question: str) -> List[str]:
        """
        Return contextually relevant follow-up question suggestions.

        Helps employees discover related policies they might not know to ask about.
        """
        follow_up_map: Dict[str, List[str]] = {
            "leave": [
                "How do I request time off in the HR system?",
                "What is the carryover policy for unused PTO?",
                "Can I take unpaid leave?",
            ],
            "benefits": [
                "When can I change my benefits elections?",
                "How do I add a dependent to my health plan?",
                "What is the employer contribution to my 401(k)?",
            ],
            "compensation": [
                "When are performance bonuses paid?",
                "How are salary reviews conducted?",
                "What is the expense reimbursement process?",
            ],
            "onboarding": [
                "What should I do on my first day?",
                "How do I set up my corporate accounts?",
                "When will I receive my equipment?",
            ],
            "offboarding": [
                "How do I transfer my projects before leaving?",
                "When will I receive my final paycheck?",
                "What happens to my unvested equity?",
            ],
            "performance": [
                "How often are performance reviews conducted?",
                "What is the promotion process?",
                "How do I set up goals for the coming year?",
            ],
        }
        return follow_up_map.get(intent, [
            "Can you clarify what type of leave this applies to?",
            "Does this policy apply to all employee types?",
            "Who should I contact for more information?",
        ])

    # ─────────────────────────────────────────────────────────────────────────
    # Escalation response builder
    # ─────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _escalation_response(
        question: str,
        reason:   str,
        intent:   str,
        employee_id: str,
        conversation_id: str,
        response_id: str,
        t0: float,
    ) -> PolicyResponse:
        """Build a PolicyResponse that routes to a human HR partner."""
        latency = (time.perf_counter() - t0) * 1000
        logger.info(
            "Escalating question  employee=%s  intent=%s  reason='%s...'",
            employee_id, intent, reason[:60],
        )
        return PolicyResponse(
            question          = question,
            answer            = (
                f"This question has been flagged for direct HR support.\n\n"
                f"**Reason:** {reason}\n\n"
                "Please contact your HR Business Partner or submit a ticket to "
                "the HR helpdesk. They will respond within 1 business day.\n\n"
                "HR Helpdesk: hr-helpdesk@company.com | Ext. 5000"
            ),
            intent            = intent,
            citations         = [],
            confidence        = 0.0,
            should_escalate   = True,
            escalation_reason = reason,
            conversation_id   = conversation_id,
            employee_id       = employee_id,
            timestamp         = datetime.utcnow().isoformat(),
            response_id       = response_id,
            follow_up_prompts = [],
            latency_ms        = latency,
        )
