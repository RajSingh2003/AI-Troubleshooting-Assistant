"""Streamlit UI for the AI Troubleshooting Assistant.

Run:  streamlit run app.py
"""
from __future__ import annotations

import html
import os

import streamlit as st

from src.assistant import DEFAULT_THRESHOLD, Answer
from src.factory import build_assistant
from src.kb import load_kb

st.set_page_config(page_title="Troubleshooting Assistant", page_icon="🛠️", layout="wide")

# (label, css class) per answer status
STATUS_BADGE = {
    "answered": ("Answered from knowledge base", "ok"),
    "need_more_info": ("Need more information", "warn"),
    "insufficient": ("Not covered by knowledge base", "bad"),
}
CONFIDENCE_CLASS = {"high": "ok", "medium": "warn", "low": "bad"}
EXAMPLES = [
    "The payment API is returning a 500 error.",
    "Customer paid via UPI, amount debited but order shows pending",
    "POS terminal at the store says no network",
    "I can't connect to VPN from home",
    "The office coffee machine is broken",
]

CSS = """
<style>
footer {visibility: hidden;}
.block-container {padding-top: 1.6rem; padding-bottom: 6rem; max-width: 1200px;}
[data-testid="stSidebar"] {background: #0F172A;}
[data-testid="stSidebar"] * {color: #E2E8F0;}
[data-testid="stSidebar"] input, [data-testid="stSidebar"] textarea {color: #0F172A !important;}
[data-testid="stSidebar"] .stButton button {
  background: #1E293B; border: 1px solid #334155; text-align: left; justify-content: flex-start;
  font-size: 0.85rem;
}
[data-testid="stSidebar"] .stButton button:hover {border-color: #3B82F6; background: #1E3A8A;}

.hero {
  background: linear-gradient(120deg, #1E3A8A 0%, #2563EB 60%, #3B82F6 100%);
  color: #fff; border-radius: 14px; padding: 22px 28px; margin-bottom: 18px;
}
.hero h1 {color: #fff; font-size: 1.65rem; margin: 0 0 4px 0; padding: 0;}
.hero p {color: #DBEAFE; margin: 0; font-size: 0.95rem;}

.kpi {background: #fff; border: 1px solid #E2E8F0; border-radius: 12px; padding: 12px 16px;}
.kpi .label {color: #64748B; font-size: 0.72rem; text-transform: uppercase; letter-spacing: .06em;}
.kpi .value {color: #0F172A; font-size: 1.05rem; font-weight: 600; margin-top: 2px;}

.pill {display: inline-block; padding: 3px 10px; border-radius: 999px; font-size: 0.78rem;
       font-weight: 600; margin-right: 6px; border: 1px solid transparent;}
.pill.ok   {background: #DCFCE7; color: #166534; border-color: #BBF7D0;}
.pill.warn {background: #FEF3C7; color: #92400E; border-color: #FDE68A;}
.pill.bad  {background: #FEE2E2; color: #991B1B; border-color: #FECACA;}
.pill.info {background: #E0E7FF; color: #3730A3; border-color: #C7D2FE;}

.summary {background: #fff; border: 1px solid #E2E8F0; border-left: 4px solid #2563EB;
          border-radius: 10px; padding: 12px 16px; margin: 10px 0 6px 0; color: #1E293B;}

.step {display: flex; gap: 14px; align-items: flex-start; background: #fff; border: 1px solid #E2E8F0;
       border-radius: 10px; padding: 12px 14px; margin-bottom: 8px;}
.step .num {flex: 0 0 28px; height: 28px; border-radius: 50%; background: #2563EB; color: #fff;
            display: flex; align-items: center; justify-content: center; font-weight: 700; font-size: 0.85rem;}
.step .text {flex: 1; color: #1E293B; line-height: 1.45;}
.kbtag {font-family: ui-monospace, monospace; font-size: 0.72rem; background: #F1F5F9; color: #475569;
        border: 1px solid #E2E8F0; border-radius: 6px; padding: 2px 7px; white-space: nowrap;}

.panel {background: #fff; border: 1px solid #E2E8F0; border-radius: 10px; padding: 14px 18px; color: #1E293B;}
.panel.ok   {border-left: 4px solid #16A34A;}
.panel.warn {border-left: 4px solid #D97706;}
.panel .title {font-size: 0.72rem; text-transform: uppercase; letter-spacing: .06em; color: #64748B; margin-bottom: 4px;}
.panel .big {font-size: 1.05rem; font-weight: 600;}

.stTabs [data-baseweb="tab-list"] {gap: 4px;}
.stTabs [data-baseweb="tab"] {padding: 6px 14px; border-radius: 8px 8px 0 0;}
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


def esc(text) -> str:
    return html.escape(str(text or ""))


# ---------------- sidebar ----------------
with st.sidebar:
    st.markdown("### 🛠️ Support Console")
    st.caption("L1/L2 troubleshooting assistant")

    st.markdown("**Connection**")
    env_key = os.getenv("GEMINI_API_KEY", "")
    typed_key = st.text_input("Gemini API key", type="password", label_visibility="collapsed",
                              placeholder="Gemini API key (optional)",
                              help="Leave blank to use GEMINI_API_KEY from .env")
    api_key = typed_key or env_key or None
    offline = st.toggle("Offline mode (keyword search only)", value=not bool(api_key))
    if not api_key and not offline:
        st.warning("No API key found. Using offline mode.")
        offline = True

    st.markdown("**Retrieval**")
    top_k = st.slider("Entries to retrieve (top-k)", 1, 5, 3)
    default_thr = DEFAULT_THRESHOLD["offline" if offline else "hybrid"]
    threshold = st.slider("Relevance threshold", 0.0, 1.0, default_thr, 0.01,
                          help="Below this score the assistant says the KB doesn't cover the issue.")
    show_debug = st.toggle("Show retrieval tab", value=True)

    if st.button("🗑️  Clear conversation", width="stretch"):
        st.session_state.messages = []
        st.rerun()

    st.divider()
    st.markdown("**Try an example**")
    for ex in EXAMPLES:
        if st.button(ex, width="stretch", key=f"ex_{ex}"):
            st.session_state.pending = ex


@st.cache_resource(show_spinner="Loading knowledge base and embeddings…")
def get_bot(api_key, offline, top_k, threshold):
    return build_assistant(api_key=api_key, offline=offline, top_k=top_k, threshold=threshold)


try:
    bot = get_bot(api_key, offline, top_k, threshold)
except Exception as exc:  # bad key, network, quota
    st.error(f"Could not initialise Gemini client: {exc}. Falling back to offline mode.")
    bot = get_bot(None, True, top_k, threshold)


# ---------------- rendering ----------------
def render_answer(ans: Answer):
    label, cls = STATUS_BADGE.get(ans.status, (ans.status, "info"))
    pills = f'<span class="pill {cls}">{esc(label)}</span>'
    if ans.status == "answered":
        pills += (f'<span class="pill {CONFIDENCE_CLASS.get(ans.confidence, "info")}">'
                  f'Confidence: {esc(ans.confidence)}</span>')
    pills += f'<span class="pill info">{"AI answer" if ans.used_llm else "KB lookup"}</span>'
    st.markdown(pills, unsafe_allow_html=True)
    if ans.summary:
        st.markdown(f'<div class="summary">{esc(ans.summary)}</div>', unsafe_allow_html=True)

    cited = [h for h in ans.hits if h.entry["id"] in ans.sources]
    sections = []
    if ans.steps:
        sections.append((f"🧭 Steps ({len(ans.steps)})", lambda: render_steps(ans)))
    if ans.resolution:
        sections.append(("✅ Resolution", lambda: render_resolution(ans)))
    if ans.escalation_team:
        sections.append(("📞 Escalation", lambda: render_escalation(ans)))
    if ans.clarifying_questions:
        sections.append(("❓ Questions", lambda: render_questions(ans)))
    if cited:
        sections.append((f"📄 Sources ({len(cited)})", lambda: render_sources(cited)))
    if show_debug and ans.hits:
        sections.append(("🔎 Retrieval", lambda: render_retrieval(ans)))
    if not sections:
        return
    for tab, (_, fill) in zip(st.tabs([s[0] for s in sections]), sections):
        with tab:
            fill()


def render_steps(ans: Answer):
    st.markdown("".join(
        f'<div class="step"><div class="num">{i}</div><div class="text">{esc(s["text"])}</div>'
        f'<span class="kbtag">{esc(s["kb_id"])}</span></div>'
        for i, s in enumerate(ans.steps, 1)), unsafe_allow_html=True)


def render_resolution(ans: Answer):
    st.markdown(f'<div class="panel ok"><div class="title">Known resolution</div>{esc(ans.resolution)}</div>',
                unsafe_allow_html=True)


def render_escalation(ans: Answer):
    when = (f'<div class="title" style="margin-top:12px">When to escalate</div>{esc(ans.escalate_when)}'
            if ans.escalate_when else "")
    st.markdown(f'<div class="panel warn"><div class="title">Escalate to</div>'
                f'<div class="big">{esc(ans.escalation_team)}</div>{when}</div>', unsafe_allow_html=True)


def render_questions(ans: Answer):
    st.markdown("To narrow this down, please tell me:")
    st.markdown("\n".join(f"- {q}" for q in ans.clarifying_questions))
    st.caption("Reply in the chat box below — your answer is combined with your earlier message.")


def render_sources(cited):
    for h in cited:
        e = h.entry
        with st.expander(f"{e['id']} — {e['title']}  ·  score {h.score:.2f}"):
            render_entry(e)


def render_retrieval(ans: Answer):
    st.caption(f"Mode: {bot.retriever.mode} · Model used: {'yes' if ans.used_llm else 'no'} "
               f"· Threshold: {bot.threshold:.2f} · Query: “{ans.retrieval_query}”")
    st.dataframe([{
        "KB": h.entry["id"], "Title": h.entry["title"],
        "Combined": round(h.score, 3),
        "Semantic": None if h.semantic is None else round(h.semantic, 3),
        "Keyword": round(h.lexical, 3),
        "Passed": "✅" if h.score >= bot.threshold else "—",
    } for h in ans.hits], hide_index=True, width="stretch")
    for n in ans.notes:
        st.caption(f"🛡️ {n}")


def render_entry(e: dict):
    st.markdown(f'<span class="pill info">{esc(e["category"])}</span>'
                f'<span class="pill warn">Severity: {esc(e.get("severity", "-"))}</span>'
                f'<span class="pill ok">Escalation: {esc(e["escalation_team"])}</span>', unsafe_allow_html=True)
    labels = ["Symptoms", "Causes", "Diagnostic checks", "Resolution"]
    t_sym, t_cause, t_chk, t_res = st.tabs(labels)
    with t_sym:
        st.markdown("\n".join(f"- {s}" for s in e["symptoms"]))
    with t_cause:
        st.markdown("\n".join(f"- {c}" for c in e.get("possible_causes", [])) or "_None listed._")
    with t_chk:
        st.markdown("\n".join(f"{i}. {c}" for i, c in enumerate(e["diagnostic_checks"], 1)))
    with t_res:
        st.markdown(e["known_resolution"])


def kpi(col, label, value):
    col.markdown(f'<div class="kpi"><div class="label">{esc(label)}</div>'
                 f'<div class="value">{esc(value)}</div></div>', unsafe_allow_html=True)


# ---------------- main ----------------
entries = load_kb()
st.markdown('<div class="hero"><h1>🛠️ AI Troubleshooting Assistant</h1>'
            '<p>For L1/L2 support. Answers come only from the knowledge base, and every step shows its source.</p>'
            '</div>', unsafe_allow_html=True)

c1, c2, c3, c4 = st.columns(4)
kpi(c1, "Knowledge base", f"{len(entries)} entries")
kpi(c2, "Mode", "Gemini + keywords" if bot.client else "Offline · keywords")
kpi(c3, "Answer model", bot.model if bot.client else "—")
kpi(c4, "Relevance threshold", f"{bot.threshold:.2f}")
st.write("")

tab_chat, tab_kb = st.tabs(["💬 Assistant", "📚 Knowledge base"])

# Outside the tabs so Streamlit pins the input to the bottom of the page.
prompt = st.chat_input("Describe the issue, e.g. “payment API returning 500”")
prompt = prompt or st.session_state.pop("pending", None)

with tab_chat:
    if "messages" not in st.session_state:
        st.session_state.messages = []

    if not st.session_state.messages and not prompt:
        st.markdown("#### How can I help?")
        st.caption("Describe the issue in the box below, or start with one of these:")
        cols = st.columns(2)
        for i, ex in enumerate(EXAMPLES[:4]):
            if cols[i % 2].button(ex, width="stretch", key=f"start_{i}"):
                st.session_state.pending = ex
                st.rerun()

    for m in st.session_state.messages:
        with st.chat_message(m["role"], avatar="🛠️" if m["role"] == "assistant" else "👤"):
            if m["role"] == "assistant" and m.get("answer") is not None:
                render_answer(m["answer"])
            else:
                st.write(m["content"])

    if prompt:
        history = list(st.session_state.messages)
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user", avatar="👤"):
            st.write(prompt)
        with st.chat_message("assistant", avatar="🛠️"):
            with st.spinner("Searching knowledge base…"):
                try:
                    ans = bot.ask(prompt, history)
                except Exception as exc:
                    st.error(f"Gemini request failed: {exc}")
                    st.stop()
            render_answer(ans)
        st.session_state.messages.append({
            "role": "assistant", "content": ans.summary, "status": ans.status, "answer": ans,
        })

with tab_kb:
    cats = sorted({e["category"] for e in entries})
    f1, f2 = st.columns([1, 1])
    pick = f1.multiselect("Filter by category", cats)
    q = f2.text_input("Search entries", placeholder="e.g. refund, VPN, 504")
    shown = [e for e in entries
             if (not pick or e["category"] in pick) and (not q or q.lower() in str(e).lower())]
    st.caption(f"Showing {len(shown)} of {len(entries)} entries · simulated data only")
    for e in shown:
        with st.expander(f"{e['id']} — {e['title']}"):
            render_entry(e)
