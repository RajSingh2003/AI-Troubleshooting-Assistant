"""Unit tests. Use a fake OpenAI client so no API key / network is needed.

Run: pytest -q
"""
import json
from types import SimpleNamespace

import pytest

from src.assistant import TroubleshootingAssistant
from src.kb import load_kb
from src.retriever import Retriever, tokenize


# ---------------- fake OpenAI client ----------------
class FakeEmbeddings:
    """Bag-of-words vectors over a fixed vocab -> deterministic cosine sim."""
    def __init__(self, vocab):
        self.vocab = vocab

    def create(self, model, input):
        data = []
        for text in input:
            toks = set(tokenize(text))
            data.append(SimpleNamespace(embedding=[1.0 if v in toks else 0.0 for v in self.vocab] + [0.01]))
        return SimpleNamespace(data=data)


class FakeChat:
    def __init__(self, payload):
        self.payload = payload
        self.last_messages = None
        self.completions = self

    def create(self, **kw):
        self.last_messages = kw["messages"]
        msg = SimpleNamespace(content=json.dumps(self.payload))
        return SimpleNamespace(choices=[SimpleNamespace(message=msg)])


def make_bot(tmp_path, payload):
    entries = load_kb()
    vocab = sorted({t for e in entries for t in tokenize(" ".join(e["keywords"] + [e["title"]]))})
    client = SimpleNamespace(embeddings=FakeEmbeddings(vocab), chat=FakeChat(payload))
    retriever = Retriever(entries, client=client, cache_path=tmp_path / "emb.json")
    return TroubleshootingAssistant(retriever, client=client, threshold=0.3), client


# ---------------- tests ----------------
def test_kb_valid():
    entries = load_kb()
    assert len(entries) >= 20
    ids = {e["id"] for e in entries}
    for e in entries:
        for r in e.get("related", []):
            assert r in ids, f"{e['id']} references missing {r}"


def test_grounded_answer_passes_validation(tmp_path):
    payload = {
        "status": "answered", "summary": "Payment API 500s",
        "steps": [{"text": "Check payment-service logs", "kb_id": "KB-001"}],
        "resolution": "Roll back", "escalation_team": "Payments Backend (L3)",
        "escalate_when": "", "clarifying_questions": [], "sources": ["KB-001"], "confidence": "high",
    }
    bot, client = make_bot(tmp_path, payload)
    ans = bot.ask("The payment API is returning a 500 error.")
    assert ans.status == "answered"
    assert ans.sources == ["KB-001"]
    assert ans.used_llm
    # Prompt actually contained the retrieved entry
    assert "[KB-001]" in client.chat.last_messages[1]["content"]


def test_hallucinated_source_is_removed_and_downgraded(tmp_path):
    payload = {
        "status": "answered", "summary": "made up",
        "steps": [{"text": "Reboot the mainframe", "kb_id": "KB-999"}],
        "resolution": "x", "escalation_team": "Imaginary Team",
        "sources": ["KB-999"], "confidence": "high",
    }
    bot, _ = make_bot(tmp_path, payload)
    ans = bot.ask("The payment API is returning a 500 error.")
    assert ans.status == "insufficient"
    assert ans.sources == [] and ans.steps == []
    assert any("KB-999" in n for n in ans.notes)


def test_invented_escalation_team_replaced(tmp_path):
    payload = {
        "status": "answered", "summary": "s",
        "steps": [{"text": "check logs", "kb_id": "KB-001"}],
        "escalation_team": "Imaginary Team", "sources": ["KB-001"],
    }
    bot, _ = make_bot(tmp_path, payload)
    ans = bot.ask("payment api 500 internal server error")
    assert ans.escalation_team == "Payments Backend (L3)"


def test_unknown_issue_skips_llm(tmp_path):
    bot, client = make_bot(tmp_path, {"status": "answered"})
    ans = bot.ask("The office coffee machine is broken")
    assert ans.status == "insufficient"
    assert not ans.used_llm
    assert client.chat.last_messages is None
    assert ans.clarifying_questions


def test_follow_up_combines_previous_query(tmp_path):
    bot, _ = make_bot(tmp_path, {"status": "need_more_info", "summary": "", "clarifying_questions": ["?"]})
    history = [
        {"role": "user", "content": "payment api failing"},
        {"role": "assistant", "content": "Which error code?", "status": "need_more_info"},
    ]
    ans = bot.ask("it's a 500", history)
    assert ans.retrieval_query.startswith("payment api failing")


def test_embedding_cache_reused(tmp_path):
    bot, _ = make_bot(tmp_path, {})
    assert (tmp_path / "emb.json").exists()
    cache = json.loads((tmp_path / "emb.json").read_text())
    assert len(cache["items"]) == len(load_kb())


def test_offline_mode_is_grounded():
    entries = load_kb()
    bot = TroubleshootingAssistant(Retriever(entries, client=None), client=None)
    ans = bot.ask("UPI payment pending, amount debited")
    assert ans.status == "answered" and ans.sources[0] == "KB-005"
    kb5 = next(e for e in entries if e["id"] == "KB-005")
    assert [s["text"] for s in ans.steps] == kb5["diagnostic_checks"]


@pytest.mark.parametrize("payload", [
    [{"status": "answered"}],                                                      # JSON list, not object
    {"status": "answered", "steps": [5], "sources": ["KB-001"]},                   # step not an object
    {"status": "answered", "steps": [{"kb_id": "KB-001"}], "sources": ["KB-001"]},  # step without text
    {"status": "answered", "steps": [{"text": "x", "kb_id": "KB-001"}], "sources": ["KB-001"],
     "escalation_team": ["Payments Backend (L3)"], "summary": None},               # wrong field types
])
def test_malformed_model_output_does_not_crash(tmp_path, payload):
    bot, _ = make_bot(tmp_path, payload)
    ans = bot.ask("The payment API is returning a 500 error.")
    assert ans.status in ("answered", "need_more_info", "insufficient")
    assert all(isinstance(s, dict) and s.get("text") for s in ans.steps)
    assert isinstance(ans.escalation_team, str) and isinstance(ans.summary, str)


def test_single_string_source_is_accepted(tmp_path):
    payload = {"status": "answered", "steps": [{"text": "check logs", "kb_id": "KB-001"}], "sources": "KB-001"}
    bot, _ = make_bot(tmp_path, payload)
    ans = bot.ask("The payment API is returning a 500 error.")
    assert ans.status == "answered" and ans.sources == ["KB-001"]
