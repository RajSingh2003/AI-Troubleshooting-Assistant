"""Troubleshooting assistant: retrieve -> gate -> grounded LLM answer -> validate."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field

from .kb import entry_context_text
from .retriever import Hit, Retriever

CHAT_MODEL = os.getenv("GEMINI_CHAT_MODEL", "gemini-3.5-flash-lite")
# Gemini 3 models are slow at low temperature (~120s vs ~1.5s per answer), so we keep the
# default temperature and limit thinking instead. Grounding is enforced by _validate().
REASONING_EFFORT = os.getenv("GEMINI_REASONING_EFFORT", "low")

# Combined-score thresholds below which we do NOT call the LLM at all.
DEFAULT_THRESHOLD = {"hybrid": 0.55, "offline": 0.25}

SYSTEM_PROMPT = """You are a troubleshooting assistant for L1/L2 support engineers.
You answer ONLY from the knowledge-base (KB) entries provided in the user message.

Rules:
1. Use only facts, checks, resolutions and escalation teams that appear in the provided KB entries.
   Never invent commands, error codes, team names, timelines or fixes that are not in the entries.
2. Decide which provided entries actually match the user's issue. Ignore entries that are only loosely related.
3. If no provided entry matches the issue, set status to "insufficient" and say the KB does not cover it.
   Do not guess a fix.
4. If an entry might match but key details are missing (e.g. which service, error code, single user vs. many),
   set status to "need_more_info" and ask 1-3 short clarifying questions.
5. Otherwise set status to "answered": give ordered troubleshooting steps taken from the entry's diagnostic checks,
   the known resolution, and the escalation team. Tag every step with the KB id it came from.
6. Be concise and practical. Write for a support engineer, not an end customer.

Return ONLY a JSON object with this shape:
{
  "status": "answered" | "need_more_info" | "insufficient",
  "summary": "one or two sentences: what the issue most likely is",
  "steps": [{"text": "step description", "kb_id": "KB-xxx"}],
  "resolution": "known resolution from the KB, or empty string",
  "escalation_team": "team from the KB, or empty string",
  "escalate_when": "when to escalate, based on the KB, or empty string",
  "clarifying_questions": ["..."],
  "sources": ["KB-xxx"],
  "confidence": "high" | "medium" | "low"
}"""


def _text(v) -> str:
    """Model output as a string: None -> "", lists joined, other values str()'d."""
    if v is None:
        return ""
    if isinstance(v, list):
        return "; ".join(_text(x) for x in v if x is not None)
    return str(v).strip()


def _as_list(v) -> list:
    """Model output as a list: a lone string/value becomes a one-item list."""
    if v is None or v == "":
        return []
    return v if isinstance(v, list) else [v]


@dataclass
class Answer:
    status: str
    summary: str
    steps: list[dict] = field(default_factory=list)
    resolution: str = ""
    escalation_team: str = ""
    escalate_when: str = ""
    clarifying_questions: list[str] = field(default_factory=list)
    sources: list[str] = field(default_factory=list)
    confidence: str = "low"
    hits: list[Hit] = field(default_factory=list)
    retrieval_query: str = ""
    used_llm: bool = False
    notes: list[str] = field(default_factory=list)


class TroubleshootingAssistant:
    def __init__(self, retriever: Retriever, client=None, top_k: int = 3,
                 threshold: float | None = None, model: str = CHAT_MODEL):
        self.retriever = retriever
        self.client = client
        self.top_k = top_k
        self.model = model
        mode = "hybrid" if client else "offline"
        self.threshold = threshold if threshold is not None else DEFAULT_THRESHOLD[mode]

    # ------------------------------------------------------------------
    def ask(self, message: str, history: list[dict] | None = None) -> Answer:
        history = history or []
        message = (message or "").strip()
        if len(message) < 3:
            return Answer(status="need_more_info",
                          summary="Please describe the issue.",
                          clarifying_questions=self._generic_questions())

        retrieval_query = self._build_retrieval_query(message, history)
        hits = self.retriever.search(retrieval_query, top_k=self.top_k)
        relevant = [h for h in hits if h.score >= self.threshold]

        # Gate 1: nothing in the KB is close enough -> do not call the LLM.
        if not relevant:
            ans = Answer(
                status="insufficient",
                summary=("The knowledge base does not contain guidance that matches this issue. "
                         "I won't guess a fix. Please add more detail, or raise it with the relevant team."),
                clarifying_questions=self._generic_questions(),
                hits=hits, retrieval_query=retrieval_query,
                notes=[f"Top retrieval score {hits[0].score:.2f} < threshold {self.threshold:.2f}"] if hits else [],
            )
            return ans

        if self.client is None:
            return self._offline_answer(relevant, hits, retrieval_query)
        return self._llm_answer(message, history, relevant, hits, retrieval_query)

    # ------------------------------------------------------------------
    @staticmethod
    def _build_retrieval_query(message: str, history: list[dict]) -> str:
        """If the assistant just asked for more info, combine the earlier
        user message with this reply so retrieval has full context."""
        if history and history[-1].get("role") == "assistant" \
                and history[-1].get("status") in ("need_more_info", "insufficient"):
            prev_user = next((m["content"] for m in reversed(history[:-1])
                              if m.get("role") == "user"), "")
            if prev_user:
                return f"{prev_user}. {message}"
        return message

    @staticmethod
    def _generic_questions() -> list[str]:
        return [
            "Which system or service is affected (e.g. payment API, UPI, POS terminal, VPN, email)?",
            "What exact error message or code do you see?",
            "Is it affecting one user/merchant or many? When did it start?",
        ]

    # ------------------------------------------------------------------
    def _llm_answer(self, message, history, relevant, hits, retrieval_query) -> Answer:
        context = "\n\n".join(entry_context_text(h.entry) for h in relevant)
        convo = ""
        recent = [m for m in history if m.get("role") in ("user", "assistant")][-4:]
        if recent:
            convo = "Recent conversation:\n" + "\n".join(
                f"{m['role']}: {m['content'][:500]}" for m in recent) + "\n\n"

        user_prompt = (
            f"{convo}"
            f"Retrieved KB entries:\n-----\n{context}\n-----\n\n"
            f"Support engineer's issue: {message}\n\n"
            "Respond with the JSON object only."
        )

        resp = self.client.chat.completions.create(
            model=self.model,
            reasoning_effort=REASONING_EFFORT,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": user_prompt},
            ],
        )
        raw = resp.choices[0].message.content or "{}"
        try:
            data = json.loads(raw)
        except json.JSONDecodeError:
            data = None
        if not isinstance(data, dict):
            data = {"status": "insufficient",
                    "summary": "The model returned an unreadable response; showing retrieved entries only."}

        ans = Answer(
            status=_text(data.get("status")) or "insufficient",
            summary=_text(data.get("summary")),
            steps=data.get("steps") if isinstance(data.get("steps"), list) else [],
            resolution=_text(data.get("resolution")),
            escalation_team=_text(data.get("escalation_team")),
            escalate_when=_text(data.get("escalate_when")),
            clarifying_questions=[_text(q) for q in _as_list(data.get("clarifying_questions")) if _text(q)],
            sources=[_text(s) for s in _as_list(data.get("sources"))],
            confidence=_text(data.get("confidence")) or "low",
            hits=hits, retrieval_query=retrieval_query, used_llm=True,
        )
        return self._validate(ans, relevant)

    def _validate(self, ans: Answer, relevant: list[Hit]) -> Answer:
        """Guardrails: the model may only cite entries we actually gave it."""
        allowed = {h.entry["id"]: h.entry for h in relevant}

        bad = [s for s in ans.sources if s not in allowed]
        if bad:
            ans.notes.append(f"Removed citations not in retrieved context: {bad}")
        ans.sources = [s for s in ans.sources if s in allowed]

        clean_steps = []
        for st in ans.steps:
            if isinstance(st, str):
                st = {"text": st, "kb_id": ""}
            if not isinstance(st, dict) or not _text(st.get("text")):
                ans.notes.append(f"Dropped malformed step: {str(st)[:60]}")
                continue
            if st.get("kb_id") in allowed:
                clean_steps.append(st)
            else:
                ans.notes.append(f"Dropped step without a valid KB source: {st.get('text', '')[:60]}")
        ans.steps = clean_steps

        # Escalation team must come from a cited entry.
        if ans.escalation_team:
            teams = {allowed[s]["escalation_team"] for s in ans.sources}
            if not any(ans.escalation_team.lower() in t.lower() or t.lower() in ans.escalation_team.lower()
                       for t in teams):
                ans.notes.append(f"Escalation team '{ans.escalation_team}' not in cited entries; replaced.")
                ans.escalation_team = "; ".join(sorted(teams))

        if ans.status == "answered" and (not ans.sources or not ans.steps):
            ans.notes.append("Answer had no valid sources/steps; downgraded to insufficient.")
            ans.status = "insufficient"
            ans.summary = ("I couldn't produce an answer grounded in the knowledge base. "
                           "Please review the retrieved entries below or provide more detail.")
            ans.steps, ans.resolution = [], ""

        if ans.status == "answered":
            top = relevant[0].score
            if top < self.threshold + 0.08 and ans.confidence == "high":
                ans.confidence = "medium"
        return ans

    # ------------------------------------------------------------------
    def _offline_answer(self, relevant, hits, retrieval_query) -> Answer:
        """No API key: return the best entry's steps verbatim (still grounded)."""
        e = relevant[0].entry
        return Answer(
            status="answered",
            summary=f"Best match in the knowledge base: {e['title']}. (Offline mode: steps shown verbatim, no AI summary.)",
            steps=[{"text": c, "kb_id": e["id"]} for c in e["diagnostic_checks"]],
            resolution=e["known_resolution"],
            escalation_team=e["escalation_team"],
            escalate_when="If the checks above do not resolve the issue.",
            # cite the best entry plus any near-equal match (avoid listing weak ones)
            sources=[h.entry["id"] for h in relevant if h.score >= 0.75 * relevant[0].score],
            confidence="medium" if relevant[0].score >= self.threshold + 0.15 else "low",
            hits=hits, retrieval_query=retrieval_query,
        )
