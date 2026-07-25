"""
analyst_chat.py – Interactive SOC Q&A interface with conversation memory.

Routes common questions locally in milliseconds using IncidentContext (no LLM required)
and falls back to LLM generation for freeform queries with context and history injected.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from copilot.utils import IncidentContext
from copilot.llm_client import LLMClient
from copilot.prompt_builder import PromptBuilder
from copilot.recommendations import RecommendationEngine

logger = logging.getLogger("Copilot.AnalystChat")


class ConversationMemory:
    """Manages chat turn history per session."""

    def __init__(self, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self.history: List[Dict[str, str]] = []  # [{"role": "user"/"assistant", "content": "..."}]

    def add_turn(self, role: str, content: str) -> None:
        self.history.append({"role": role, "content": content})
        if len(self.history) > self.max_turns * 2:
            self.history = self.history[-self.max_turns * 2:]

    def get_history_string(self) -> str:
        """Serializes history as flat text for prompt inclusion."""
        lines = []
        for turn in self.history:
            role = "Analyst" if turn["role"] == "user" else "Copilot"
            lines.append(f"{role}: {turn['content']}")
        return "\n".join(lines)

    def clear(self) -> None:
        self.history.clear()


class AnalystChat:
    """Handles analyst queries grounded strictly in IncidentContext."""

    def __init__(self, llm_client: LLMClient) -> None:
        self.llm = llm_client
        self.memory_store: Dict[str, ConversationMemory] = {}
        self.rec_engine = RecommendationEngine()

    def get_memory(self, session_id: str) -> ConversationMemory:
        if session_id not in self.memory_store:
            self.memory_store[session_id] = ConversationMemory()
        return self.memory_store[session_id]

    def ask(self, ctx: IncidentContext, question: str) -> str:
        """Process an analyst question.

        Parameters
        ----------
        ctx : IncidentContext
        question : str

        Returns
        -------
        str  Copilot's answer.
        """
        memory = self.get_memory(ctx.session_id)
        
        # 1. Check local pattern-matching routes (fast, deterministic, grounded)
        answer = self._try_route_locally(ctx, question)
        
        if answer is None:
            # 2. Fall back to LLM query with context & conversation history
            history_str = memory.get_history_string()
            prompt = PromptBuilder.build_prompt(
                "analyst_qa", ctx, question=question, history=history_str
            )
            system = PromptBuilder.build_system_prompt()
            
            try:
                answer = self.llm.generate(prompt=prompt, system=system)
            except Exception as e:
                logger.error("LLM Q&A generation failed: %s. Using template fallback.", e)
                answer = (
                    "**[Template Mode Fallback]** I was unable to reach the AI engine. "
                    "Please ask a direct structured question (e.g. 'what should I investigate', 'mitre tactic', or 'top features')."
                )

        # Update memory
        memory.add_turn("user", question)
        memory.add_turn("assistant", answer)
        
        return answer

    def _try_route_locally(self, ctx: IncidentContext, question: str) -> Optional[str]:
        """Routes common incident queries directly using structured fields."""
        q = question.lower().strip()

        # Prediction explanation / classification
        if any(w in q for w in ["why was this classified", "what was the classification", "why is it classified"]):
            if ctx.attack_type == "Normal":
                return f"Session {ctx.session_id} was classified as Normal because its behavioral indicators did not cross the anomaly threshold."
            top_pos = [f"'{c['feature']}'" for c in ctx.positive_contributors[:3]]
            return (
                f"Session {ctx.session_id} was classified as **{ctx.attack_type}** with {ctx.confidence:.0%} confidence. "
                f"The classification was driven by key feature deviations: {', '.join(top_pos)}. "
                f"Baseline text explanation:\n\n{ctx.nl_explanation}"
            )

        # Top features
        if any(w in q for w in ["which features contributed", "top features", "contributing features", "most important feature"]):
            if not ctx.positive_contributors:
                return "There are no positive feature deviations recorded for this session."
            lines = [f"- **{c['feature']}**: value={c['value']}, impact={c['impact']} (SHAP={c['shap_value']:.4f})" 
                     for c in ctx.positive_contributors[:5]]
            return "The top contributing behavioral anomalies for this session are:\n" + "\n".join(lines)

        # MITRE ATT&CK
        if "mitre" in q:
            if not ctx.mitre:
                return "This session is classified as Normal and has no associated MITRE ATT&CK mapping."
            return (
                f"This incident corresponds to the following MITRE ATT&CK matrix:\n"
                f"- **Tactic:** {ctx.mitre_tactic}\n"
                f"- **Technique ID:** {ctx.mitre_technique_id}\n"
                f"- **Technique Name:** {ctx.mitre_technique}"
            )

        # Investigation checklist
        if any(w in q for w in ["what should i investigate", "investigation checklist", "investigate first"]):
            if not ctx.investigation_steps:
                return "No investigation recommendations are recorded for this baseline session."
            lines = [f"{i+1}. {step}" for i, step in enumerate(ctx.investigation_steps)]
            return "Recommended Investigation Checklist:\n" + "\n".join(lines)

        # Severity
        if any(w in q for w in ["why is the severity", "severity level", "what is the severity"]):
            return (
                f"The severity is classified as **{ctx.severity}**. "
                f"This categorization is derived from the session risk score of **{ctx.risk_score:.2f}** "
                f"and sequence reconstruction anomaly score of **{ctx.anomaly_score:.4f}**."
            )

        # Business Impact
        if "business impact" in q:
            from copilot.utils import BUSINESS_IMPACT
            return f"**Potential Business Impact:**\n{BUSINESS_IMPACT.get(ctx.attack_type, 'No impact details.')}"

        # Containment
        if any(w in q for w in ["containment", "what should i do", "how to contain"]):
            recs = self.rec_engine.generate(ctx)
            lines = [f"- [ ] {a}" for a in recs.immediate_containment]
            return (
                f"**Immediate Containment Playbook:**\n"
                f"Priority Action: *{recs.priority_action}*\n\n"
                f"Playbook Steps:\n" + "\n".join(lines)
            )

        # Confidence
        if "confidence" in q:
            top3 = ", ".join([f"{p['attack']} ({p['probability']:.0%})" for p in ctx.top3_predictions])
            return (
                f"The classifier has a confidence score of **{ctx.confidence:.1%}** for the prediction '{ctx.attack_type}'. "
                f"The top predictions generated by the model are: {top3}."
            )

        # Clear memory
        if q in ("clear", "clear history", "reset"):
            self.get_memory(ctx.session_id).clear()
            return "Conversation history cleared."

        return None  # Route to LLM
