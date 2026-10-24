# AI Troubleshooting Assistant for L1/L2 Support

A working prototype that helps support engineers investigate common technical issues. An engineer describes a problem in plain language. The assistant finds the matching entries in a sample knowledge base (KB), suggests step-by-step troubleshooting checks, names the team to escalate to, and shows the KB entry behind every step. If the KB does not cover the issue, it says so and asks for more detail instead of inventing a fix.

> **All data is simulated.** No production systems, real customers or real credentials are used.

---

## Contents

1. [Features](#1-features)
2. [Setup](#2-setup)
3. [Project structure](#3-project-structure)
4. [Approach](#4-approach)
5. [AI tools and models used](#5-ai-tools-and-models-used)
6. [My contribution and use of AI coding tools](#6-my-contribution-and-use-of-ai-coding-tools)
7. [Assumptions](#7-assumptions)
8. [Testing](#8-testing)
9. [Limitations](#9-limitations)
10. [Possible next steps](#10-possible-next-steps)
11. [Assignment requirements checklist](#11-assignment-requirements-checklist)

---

## 1. Features

- **Plain-language chat interface** (Streamlit) with example queries and follow-up questions.
- **Hybrid retrieval:** semantic embeddings plus keyword matching, so both paraphrases ("card machine not connecting") and exact error codes (`500`, `429`, `51`) are found.
- **Grounded AI answers:** the model may only use the retrieved KB entries, and every step is tagged with its KB id (for example `[KB-001]`).
- **Escalation guidance:** the likely escalation team and when to escalate, both taken from the KB.
- **Source transparency:** the cited KB entries open in place, and a retrieval tab shows the scores and the guardrail decisions.
- **Responsible handling of unknown issues:** if nothing is relevant enough, the model is not called, and the assistant says the KB is insufficient and asks clarifying questions.
- **Offline mode:** without an API key, the app still works using keyword search and shows KB steps word for word.

---

## 2. Setup

**Requirements:** Python 3.10+ and, optionally, a free Google Gemini API key from https://aistudio.google.com/apikey.

```bash
python -m venv .venv
source .venv/bin/activate          # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_key_here

# Optional overrides
# GEMINI_CHAT_MODEL=gemini-3.5-flash-lite
# GEMINI_REASONING_EFFORT=low
# GEMINI_EMBED_MODEL=gemini-embedding-001
```

Run the app:

```bash
streamlit run app.py
```

Open http://localhost:8501. You can also paste the key into the sidebar instead of using `.env`.

**No key?** The app starts in **offline mode**: keyword retrieval only, and steps are shown exactly as written in the KB.

---

## 3. Project structure

```
troubleshooting/
├── app.py                    # Streamlit UI (chat, sources, retrieval debug, KB browser)
├── src/
│   ├── kb.py                 # KB loading, schema validation, text for search / prompt
│   ├── retriever.py          # Hybrid retriever + embedding cache
│   ├── assistant.py          # Gates, system prompt, LLM call, validation, offline mode
│   └── factory.py            # Builds the client/assistant from .env (Gemini endpoint)
├── kb/knowledge_base.json    # 25 simulated KB entries
├── data/embeddings.json      # Cached entry embeddings (auto-generated)
├── tests/
│   ├── test_assistant.py     # 13 unit tests (mock client, no key needed)
│   ├── test_queries.json     # 30 evaluation queries (25 in-KB, 5 out-of-KB)
│   └── evaluate.py           # Evaluation script
├── .streamlit/config.toml    # UI theme
├── requirements.txt
└── README.md
```

---

## 4. Approach

### 4.1 Problem understanding

L1/L2 engineers spend a lot of time searching documentation for known symptoms, checks and fixes. An AI assistant can speed this up, but in support work **a confident wrong answer is worse than no answer**. An invented fix can cause an outage, a wrong escalation wastes another team's time, and a made-up "known resolution" damages trust in the tool.

So the main design goal was:

> **Be helpful when the knowledge base has the answer, and be honest when it doesn't.**

| # | Success criterion | How it is measured |
|---|---|---|
| 1 | Finds the right KB entry for issues the KB covers | Retrieval accuracy on 25 in-KB queries |
| 2 | Answers are grounded and show their source | Every step carries a valid KB id, enforced in code |
| 3 | Never invents a fix when the KB is insufficient | 5 out-of-KB queries must return "insufficient", plus unit tests with fake hallucinations |
| 4 | Gives practical, ordered steps and an escalation team | Fixed response schema with `steps`, `resolution`, `escalation_team`, `escalate_when` |
| 5 | Easy to demo and run | Streamlit UI, offline mode without an API key |

### 4.2 Solution overview

The assistant follows a **Retrieval-Augmented Generation (RAG)** pattern with two safety gates:

```
User question
   │
   ▼
1. Hybrid retrieval ──► top-k KB entries with scores
   (Gemini embeddings 70% + keyword match 30%)
   │
   ▼
2. Gate 1 – relevance threshold
   best score < threshold? ──► "Not covered by KB" + clarifying questions (no model call)
   │
   ▼
3. Grounded model call (gemini-3.5-flash-lite, JSON output)
   system prompt: answer ONLY from the given entries, cite a KB id per step,
   return need_more_info / insufficient if the entries don't fit
   │
   ▼
4. Gate 2 – validation in code
   - remove citations to entries that were not retrieved
   - drop steps without a valid KB id
   - escalation team must come from a cited entry
   - "answered" with no valid sources → downgraded to "insufficient"
   │
   ▼
5. UI: status badge, steps tagged [KB-xxx], resolution, escalation team,
       expandable source entries, retrieval score table
```

The key idea is **defence in depth**: the prompt *asks* the model to stay grounded, and the code *checks* that it did. A model can ignore a prompt, but it can't get past the validation code.

Each answer has one of three statuses:

| Status | Meaning |
|---|---|
| `answered` | A KB entry matches. Steps, resolution and escalation team are shown with sources. |
| `need_more_info` | An entry may match, but key details are missing. The assistant asks 1–3 clarifying questions. |
| `insufficient` | The KB does not cover the issue. No fix is suggested. |

### 4.3 Knowledge base design

`kb/knowledge_base.json` has **25 entries** in four areas of a payments / fintech support desk:

| Area | Entries |
|---|---|
| Payments (API, UPI, cards, refunds, settlement, recon) | KB-001 – KB-011, KB-013 |
| POS terminals | KB-014 – KB-017 |
| Infrastructure (DB, SSL, CPU/memory, disk) | KB-012, KB-019 – KB-021 |
| Access / IT support (merchant login, AD, VPN, email, laptop) | KB-018, KB-022 – KB-025 |

- **Schema:** `id, title, category, symptoms, keywords, possible_causes, diagnostic_checks, known_resolution, escalation_team, severity, related`. It follows how support runbooks are usually written: *what you see → why it happens → what to check → how to fix → who owns it*.
- **Deliberate gaps:** Kafka, Microsoft Teams, the HR portal and office equipment are left out on purpose, so the "unknown issue" behaviour can be tested.
- **Validation at load time:** required fields and unique ids are checked when the app starts, so a broken entry fails early.
- **Two text views per entry:**
  - *Search text* (title, symptoms, keywords, causes) is used for retrieval. It describes **the problem**, because that is how users phrase questions.
  - *Context text* (the full entry, including checks, resolution and escalation) is what the model receives as grounding.
- **Adding an entry:** add an object with a new `id` to the JSON. On the next start, only new or changed entries are re-embedded; the rest come from the cache in `data/embeddings.json`.

### 4.4 Hybrid retrieval

| Method | Good at | Weak at |
|---|---|---|
| Semantic (embeddings) | Paraphrases: "card machine not connecting" ≈ "POS terminal offline" | Exact codes: `500` vs `502` vs `504` can look similar |
| Lexical (keywords) | Exact error codes and product terms (`429`, `UPI`, `3DS`) | Synonyms and natural wording |

The two are combined:

```
score = 0.7 × cosine_similarity(query, entry) + 0.3 × keyword_score(query, entry)
```

- **Keyword score:** query words (stopwords removed) are matched against **strong** fields (title, keywords) at full weight and **weak** fields (symptoms, causes, category) at half weight. Matching an exact keyword phrase (for example "internal server error") adds a bonus. The score is capped at 1.0.
- **In-memory search** instead of a vector database: with 25 entries a linear scan takes milliseconds, and a vector database would add setup with no benefit.
- **Embedding cache:** entry embeddings are saved with a content hash, so startup is fast and API usage stays low.
- **Offline fallback:** without an API key, retrieval is keyword-only with its own threshold (0.25), so the demo never depends on the network.

### 4.5 Gate 1: relevance threshold (before the model)

If the best retrieval score is below the threshold (0.55 in hybrid mode), **the model is not called at all**. The assistant replies that the KB does not cover the issue and asks clarifying questions (which system, what error, how many users affected).

- **No chance to hallucinate:** if the model never sees the question, it can't invent an answer.
- **Cheaper and faster:** out-of-scope questions cost no model tokens.
- **Explainable:** the retrieval tab shows the score and the threshold, so an engineer can see *why* it refused.

**How 0.55 was chosen:** in the evaluation set, in-KB queries scored 0.70–0.90 and out-of-KB queries 0.38–0.44 with Gemini embeddings. 0.55 sits in the gap. The earlier value (0.35) let off-topic queries through; raising it moved retrieval accuracy from 25/30 to 30/30. The threshold can be adjusted with a sidebar slider.

### 4.6 Prompt engineering

The system prompt (`src/assistant.py`) is short and rule-based. Each rule targets a specific failure:

| Rule in the prompt | Failure it prevents |
|---|---|
| "Answer ONLY from the KB entries provided" | Using general knowledge that may not apply to this organisation |
| "Never invent commands, error codes, team names, timelines or fixes" | The most common hallucinations in support answers |
| "Ignore entries that are only loosely related" | Mixing in steps from a nearby but wrong entry |
| "If no entry matches → `insufficient`" | Guessing a fix for an unknown issue |
| "If key details are missing → `need_more_info` + 1–3 questions" | Answering an ambiguous question too early |
| "Tag every step with the KB id it came from" | Steps that can't be traced to a source |
| "Write for a support engineer, be concise" | Long, customer-style answers |

Other prompt choices:
- **Structured JSON output** (`response_format = json_object`) with a fixed schema: `status, summary, steps[{text, kb_id}], resolution, escalation_team, escalate_when, clarifying_questions, sources, confidence`. The UI and validation code rely on fields instead of parsing free text.
- **Three explicit statuses** give the model an approved way to say "I don't know". Without one, models tend to answer anyway.
- **Context layout:** retrieved entries sit between `-----` markers, each starting with `[KB-xxx]`, so the model can quote ids exactly.
- **Short conversation memory:** the last 4 messages (each cut to 500 characters) are included so follow-ups make sense.
- **Follow-ups:** when the assistant asked for more information, the next message is combined with the earlier one for retrieval ("payment API failing" + "it's a 500" becomes one query).
- **Model settings:** reasoning effort `low` and default temperature. A low temperature (0.1) made answers take about 120 s with this model family; the current settings take about 1.5 s. Grounding is enforced by code, not by temperature, so this trade-off is safe.

### 4.7 Gate 2: validation in code (after the model)

The model's output is treated as **untrusted** until it passes these checks (`_validate()` in `src/assistant.py`):

1. **Citation check:** source ids that were not retrieved (for example `KB-999`) are removed.
2. **Step check:** each step needs text and a `kb_id` from the retrieved set; others are dropped.
3. **Escalation check:** the escalation team must match a cited entry; an invented team is replaced with the KB value.
4. **Downgrade rule:** an `answered` response with no valid sources or steps becomes `insufficient`, and its resolution is cleared.
5. **Confidence cap:** if the top score is only just above the threshold, `high` confidence is lowered to `medium`.
6. **Robust parsing:** malformed output (a list instead of an object, wrong field types, invalid JSON) is normalised or safely rejected instead of crashing.

Every intervention is shown in the UI with a 🛡️ note, so engineers can see when and why the guardrails acted.

### 4.8 Handling unknown and ambiguous issues

| Situation | Behaviour |
|---|---|
| Very short or empty input | `need_more_info`: "Please describe the issue." |
| Nothing relevant in the KB (score < threshold) | `insufficient` + clarifying questions, **no model call** |
| An entry might match but details are missing | `need_more_info` + 1–3 specific questions |
| Model cites sources it was not given | Citations removed; answer downgraded if nothing valid remains |
| User replies to a clarifying question | Previous and new messages are merged for retrieval |

### 4.9 Worked example

**Input:** "The payment API is returning a 500 error."

1. **Retrieval:** KB-001 ("Payment API returns HTTP 500 Internal Server Error") ranks first: the embeddings match the meaning, and the keywords match "payment api" and "500".
2. **Gate 1:** the score is above 0.55, so the request continues.
3. **Model:** returns `answered` with steps tagged `[KB-001]`: check payment-service logs, check recent deployments, check DB connection pool metrics, check acquirer / bank gateway health, confirm whether all merchants or one are affected.
4. **Resolution:** roll back a bad deployment; if the DB pool is exhausted, follow KB-012; if a gateway is failing, enable failover.
5. **Escalation:** Payments Backend (L3).
6. **Gate 2:** all citations are valid, so nothing is changed. KB-001 can be opened from the Sources tab.

**Counter-example:** "Kafka consumer lag is increasing on the orders topic" → the top score is far below the threshold → **"Not covered by the knowledge base"** with clarifying questions. No fix is suggested and the model is not called.

### 4.10 Trade-offs

| Decision | Benefit | Cost |
|---|---|---|
| Strict grounding + validation | Trustworthy, traceable answers | May refuse questions a general model could answer |
| Threshold gate before the model | No hallucination on unknown topics, lower cost | Needs tuning; a badly worded valid query may be rejected |
| Lighter, faster model (`flash-lite`, low reasoning) | ~1.5 s answers, good for a live demo | Slightly less reliable than a larger model (29/30) |
| In-memory search, JSON KB | Simple and transparent | Will not scale to thousands of entries |
| Gemini via the OpenAI-compatible SDK | Easy to switch providers | Depends on the compatibility layer |
| Offline mode | Demo works without a key or network | Keyword-only search is less flexible |

---

## 5. AI tools and models used

| Purpose | Model / tool |
|---|---|
| Embeddings | Google `gemini-embedding-001` |
| Answer generation | Google `gemini-3.5-flash-lite`, reasoning effort `low` (change with `GEMINI_CHAT_MODEL` / `GEMINI_REASONING_EFFORT`) |
| API client | `openai` Python SDK pointed at Gemini's OpenAI-compatible endpoint |
| UI | Streamlit |
| AI coding assistant | Claude Code (see section 6) |

---

## 6. My contribution and use of AI coding tools

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

## 7. Assumptions

- The users are support engineers, so answers are technical and short.
- The KB is small (about 25 entries), so search runs in memory with no vector database.
- Escalation teams, services and system names are generic or made up.
- The KB is the single source of truth: if it is wrong, the answer will be wrong too.

---

## 8. Testing

```bash
pytest -q                                   # unit tests (mock client, no key needed)
python -m tests.evaluate --offline          # 30 queries, keyword retrieval only
python -m tests.evaluate --retrieval-only   # 30 queries, hybrid retrieval (embeddings only, cheap)
python -m tests.evaluate                    # full pipeline including model answers
```

**Unit tests:** 13 tests in `tests/test_assistant.py` use a fake OpenAI-compatible client, so they are fast, free and deterministic. They focus on the guardrails rather than the model's wording, and check that:
- grounded answers pass validation and the prompt contains the retrieved entry
- a made-up citation (`KB-999`) is removed and the answer is downgraded
- a made-up escalation team is replaced with the KB value
- unknown issues never call the model
- follow-up queries are merged
- the embedding cache works
- offline answers match the KB exactly
- malformed model output (wrong JSON shape or field types) is cleaned up instead of crashing

**Evaluation set:** `tests/test_queries.json` has 25 in-KB queries (one per entry, in natural language) and 5 out-of-KB queries (coffee machine, HR portal, Kafka, Teams, "something is broken").

| Mode | Result |
|---|---|
| Unit tests | **13/13 pass** |
| Offline keyword retrieval | **30/30** |
| Hybrid retrieval (Gemini embeddings) | **30/30** |
| Full pipeline with `gemini-3.5-flash-lite` | **29/30** |

The one full-pipeline miss was the Outlook query, which returned "insufficient" in one run; 8 reruns all answered correctly. This shows LLM variance, and also that the system **fails safe**: when unsure, it refused instead of inventing an answer.

The free tier allows 15 answer requests per minute, so the full run may pause or hit a 429 near the end.

---

## 9. Limitations

- **Threshold tuning:** the default hybrid threshold (0.55) was tuned on the 30-query evaluation set. Adjust it with the sidebar slider if valid queries are rejected or unrelated ones get through.
- **Evaluation bias:** the same person wrote the KB and the evaluation queries, so real users will word things in ways that score lower.
- **No live system access:** the prototype is not connected to logs, monitoring or ticketing, so it suggests checks but can't run them.
- **Wording drift:** the model can reword a step slightly. Validation checks which entry a step cites, not every word.
- **Short memory:** only the last 4 messages are used as conversation context.
- **KB quality:** an outdated resolution in the KB will be returned with confidence.
- **Model variance:** LLM output is not fully deterministic, as the 29/30 full-pipeline result shows.
- **External API dependency:** rate limits and temporary 503 errors are handled with retries but can still slow things down.

---

## 10. Possible next steps

- **KB management:** owners, review dates and versioning for entries, with stale entries flagged.
- **Feedback loop:** record helpful / not helpful for each answer, and use `insufficient` queries to decide which KB entries to write next.
- **Scale retrieval:** a vector database and re-ranker once the KB has hundreds or thousands of entries.
- **Read-only integrations:** monitoring status (for example "is payment-service healthy right now?"), recent deployments, ticket history.
- **Ticket hand-off:** create the escalation ticket with steps tried and sources pre-filled.
- **Security:** SSO, role-based KB access, PII redaction before text is sent to the model, and audit logs.
- **Continuous evaluation:** run the evaluation set in CI on every KB or prompt change.

---

## 11. Assignment requirements checklist

| Requirement | Where it is met |
|---|---|
| Sample KB with issues, symptoms, checks and resolutions | `kb/knowledge_base.json` (25 entries), section 4.3 |
| Chat/search interface in plain language | `app.py` (Streamlit chat) |
| Retrieve relevant entries and use AI to write the response | `src/retriever.py`, `src/assistant.py`, sections 4.4 and 4.6 |
| Step-by-step checks and likely escalation team | Steps, Resolution and Escalation tabs in the UI |
| Show the KB entry or source used | `[KB-xxx]` tag on every step, plus the Sources tab |
| Handle unknown issues responsibly | Gate 1, `need_more_info` / `insufficient` statuses, Gate 2 (sections 4.5, 4.7, 4.8) |
| Source code and setup instructions | This repository, section 2 |
| README: approach, AI tools/models, assumptions, testing, limitations | Sections 4, 5, 7, 8 and 9 |