# AI Troubleshooting Assistant for L1/L2 Support

A small prototype for support engineers. You describe a technical issue in plain language, and it finds matching entries in a sample knowledge base (KB). It then uses Google Gemini to suggest step-by-step checks and the team to escalate to, and shows which KB entry each step came from. If the KB doesn't cover the issue, it says so instead of guessing.

> All data is simulated. No production systems are used.

---

## 1. Setup

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# put your Gemini API key in .env as GEMINI_API_KEY=...
streamlit run app.py
```

Open http://localhost:8501. You can also paste a key in the sidebar instead of using `.env`.
**No key?** The app runs in *offline mode*: keyword search only, and steps are shown word for word from the KB.

Run tests:

```bash
pytest -q                               # unit tests (mock client, no key needed)
python -m tests.evaluate --offline      # 30 sample queries, keyword retrieval
python -m tests.evaluate --retrieval-only   # same, with Gemini embeddings (cheap)
python -m tests.evaluate                # full pipeline incl. model answers
```

---

## 2. Approach

```
User question
   │
   ▼
Hybrid retrieval ──► top-k KB entries with scores
 (Gemini embeddings 70% + keyword match 30%)
   │
   ▼
Gate 1: best score < threshold? ──► "Not covered by KB" + clarifying questions (no model call)
   │
   ▼
Grounded model call (gemini-3.5-flash-lite, reasoning effort low, JSON output)
 system prompt: answer ONLY from given entries; cite KB id per step;
 return need_more_info / insufficient if entries don't fit
   │
   ▼
Gate 2: validation in code
 - remove citations to entries that weren't retrieved
 - drop steps without a valid KB id
 - escalation team must come from a cited entry
 - "answered" with no valid sources → downgraded to "insufficient"
   │
   ▼
UI: status badge, steps tagged [KB-xxx], resolution, escalation team,
    expandable source entries, retrieval score table
```

**Why hybrid retrieval?** Embeddings handle paraphrases ("card machine not connecting" vs. "POS terminal offline"). Keyword matching catches exact error codes (`500`, `429`, `51`) that embeddings can blur together.

**Why two gates?** The prompt asks the model to stay grounded, and the code then checks that it did. A model can ignore a prompt, but it can't get past the validation code.

**Follow-up questions:** if the assistant asked for more information, your next message is combined with your previous one for retrieval. For example, "payment API failing" followed by "it's a 500" becomes one query.

---

## 3. Knowledge base

`kb/knowledge_base.json` has 25 entries covering three areas:

| Area | Entries |
|---|---|
| Payments (API, UPI, cards, refunds, settlement, recon) | KB-001 – KB-011, KB-013 |
| POS terminals | KB-014 – KB-017 |
| Infrastructure (DB, SSL, CPU/memory, disk) | KB-012, KB-019 – KB-021 |
| Access / IT support (merchant login, AD, VPN, email, laptop) | KB-018, KB-022 – KB-025 |

Fields in each entry: `id, title, category, symptoms, keywords, possible_causes, diagnostic_checks, known_resolution, escalation_team, severity, related`.

Some topics are left out on purpose (Kafka, Teams, HR portal, office equipment) so the "unknown issue" behaviour can be tested.

**Adding an entry:** add an object to the JSON with a new `id`. On the next start, only new or changed entries are re-embedded; the rest are cached in `data/embeddings.json`.

---

## 4. AI tools and models

| Purpose | Model |
|---|---|
| Embeddings | Google `gemini-embedding-001` |
| Answer generation | Google `gemini-3.5-flash-lite`, reasoning effort `low` (change with `GEMINI_CHAT_MODEL` / `GEMINI_REASONING_EFFORT`) |
| Client | `openai` Python SDK pointed at Gemini's OpenAI-compatible endpoint |
| UI | Streamlit |

### My contribution and use of AI coding tools

**Initial prototype (my own work).** I designed and wrote the first version myself: the 25-entry knowledge base and its schema, the hybrid retriever (embeddings + keyword scoring with an embedding cache), the two-gate design (score threshold before the model call, citation/escalation validation after it), the Streamlit UI, the unit tests with a mock client, and the 30-query evaluation set including deliberate out-of-KB questions. AI coding tools were not used for this version.

**Changes made with Claude Code (AI coding assistant).** I directed these changes and reviewed the results; Claude Code wrote the code and ran the tests:

| Change | Why | How it was checked |
|---|---|---|
| Switched from OpenAI to Google Gemini (OpenAI-compatible endpoint, `GEMINI_API_KEY`) | My OpenAI account had no credits; I chose Gemini | Live calls with my Gemini key |
| Raised the relevance threshold from 0.35 to 0.55 | Gemini embeddings score higher, so off-topic questions (coffee machine, HR portal, Kafka) were getting answers | Retrieval evaluation went from 25/30 to 30/30 |
| Replaced `gemini-2.5-flash` with a current model | Google returned "no longer available to new users" | Live call |
| Cleaned up the model's JSON before validation | A list instead of an object, wrong field types, or steps without text crashed the app or rejected valid answers | 5 new unit tests (13/13 pass) |
| More retries on the API client | Gemini returned temporary 503 "high demand" errors | — |
| `gemini-3.5-flash-lite`, reasoning effort low, default temperature | Answers took ~100 s; temperature 0.1 was the main cause | Measured: ~120 s → ~1.5 s per answer; full evaluation 29/30 |

My decisions: which provider to use, accepting the faster/lighter model as the speed–quality trade-off, and asking for a bug review of the whole project.

---

## 5. Testing

- **Unit tests** (`tests/test_assistant.py`, 13 tests) use a mock client and check:
  - grounded answers pass validation
  - a made-up citation (`KB-999`) is removed and the answer is downgraded
  - a made-up escalation team is replaced with the KB value
  - unknown issues never call the model
  - follow-up queries are merged
  - the embedding cache works
  - offline answers match the KB exactly
  - malformed model output (wrong JSON shape or field types) is cleaned up instead of crashing
- **Evaluation set** (`tests/test_queries.json`): 25 in-KB queries and 5 out-of-KB queries.
  Offline keyword retrieval: **30/30**. Hybrid retrieval (Gemini embeddings): **30/30**. Full pipeline with `gemini-3.5-flash-lite`: **29/30** (one run returned "insufficient" for the Outlook query; 8 reruns all answered correctly). The free tier allows 15 answer requests per minute, so the full run may pause or hit a 429 near the end.

---

## 6. Assumptions

- Users are support engineers, so answers are technical and short.
- The KB is small (about 25 entries), so search runs in memory with no vector database.
- Escalation teams and system names are generic or made up.

## 7. Limitations

- **Threshold tuning:** the default hybrid threshold (0.55) was tuned on the 30-query evaluation set with Gemini embeddings (in-KB queries scored 0.70–0.90, out-of-KB 0.38–0.44). Run `python -m tests.evaluate --retrieval-only` with your key and adjust it with the sidebar slider if valid queries are rejected or unrelated ones get through.
- The evaluation queries were written by the same person as the KB, so real users will word things in ways that score lower.
- The prototype doesn't connect to live logs, monitoring or ticketing, so it can't check the system's current state.
- The model can still reword a step slightly. Validation checks which entry a step cites, not every word of it.
- Conversation memory is short (the last 4 messages).
- A KB entry is only as good as its author; an outdated resolution will be returned with confidence.

## 8. Possible next steps

- Log feedback (helpful / not helpful) for each answer, and use unanswered queries to decide which KB entries to write next.
- Use a vector store and re-ranker if the KB grows to thousands of entries.
- Connect read-only to monitoring (for example, "is payment-service healthy right now?").
