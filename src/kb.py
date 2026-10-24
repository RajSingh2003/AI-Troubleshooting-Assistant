"""Knowledge base loading and text preparation."""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
KB_PATH = ROOT / "kb" / "knowledge_base.json"

REQUIRED_FIELDS = [
    "id", "title", "category", "symptoms", "keywords",
    "diagnostic_checks", "known_resolution", "escalation_team",
]


def load_kb(path: Path | str = KB_PATH) -> list[dict]:
    """Load and validate KB entries."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    entries = data["entries"] if isinstance(data, dict) else data
    seen = set()
    for e in entries:
        missing = [f for f in REQUIRED_FIELDS if not e.get(f)]
        if missing:
            raise ValueError(f"KB entry {e.get('id', '?')} missing fields: {missing}")
        if e["id"] in seen:
            raise ValueError(f"Duplicate KB id: {e['id']}")
        seen.add(e["id"])
    return entries


def entry_search_text(e: dict) -> str:
    """Text that represents an entry for embedding / search.

    Focuses on what a user would describe (title, symptoms, keywords, causes)
    rather than the resolution, so queries match on the problem.
    """
    parts = [
        e["title"],
        f"Category: {e['category']}",
        "Symptoms: " + "; ".join(e["symptoms"]),
        "Keywords: " + ", ".join(e["keywords"]),
    ]
    if e.get("possible_causes"):
        parts.append("Possible causes: " + "; ".join(e["possible_causes"]))
    return "\n".join(parts)


def entry_context_text(e: dict) -> str:
    """Full entry text passed to the LLM as grounding context."""
    lines = [
        f"[{e['id']}] {e['title']}",
        f"Category: {e['category']}",
        f"Severity: {e.get('severity', 'n/a')}",
        "Symptoms:",
        *[f"  - {s}" for s in e["symptoms"]],
    ]
    if e.get("possible_causes"):
        lines += ["Possible causes:", *[f"  - {c}" for c in e["possible_causes"]]]
    lines += [
        "Diagnostic checks:",
        *[f"  {i}. {c}" for i, c in enumerate(e["diagnostic_checks"], 1)],
        f"Known resolution: {e['known_resolution']}",
        f"Escalation team: {e['escalation_team']}",
    ]
    if e.get("related"):
        lines.append("Related entries: " + ", ".join(e["related"]))
    return "\n".join(lines)
