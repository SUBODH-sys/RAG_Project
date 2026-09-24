"""
Blissbot — Mental Health Education Chatbot
Streamlit app wired to a ChromaDB + BGE-small + Groq RAG pipeline,
with a crisis-detection guardrail layer.

Run:
    streamlit run app.py

Requirements (put in requirements.txt):
    streamlit
    chromadb
    sentence-transformers
    langchain-groq
    openai
    python-dotenv

Secrets:
    Local:  create a .env file with:
                GROQ_API_KEY=gsk_...
                OPENAI_API_KEY=sk-...      (used ONLY for the free moderation
                                             endpoint, no chat completions)
    Cloud:  Streamlit Cloud -> App settings -> Secrets, same two keys.

    OPENAI_API_KEY is optional: if it's missing, the app falls back to
    regex-only crisis detection rather than crashing. You'll see a one-time
    warning in the sidebar if that happens.
"""

import os
import re
import time
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# ---------------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------------

PERSIST_DIR = "data/vector_store"          # same path your notebook used
COLLECTION_NAME = "json_qa_documents"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GROQ_MODEL = "qwen/qwen3.8-27b"               
TOP_K = 3

# US-centric by default since your dataset's `region` field is "US" — adjust/expand
# per-region resources if you localize this later.
CRISIS_RESOURCES = {
    "US": [
        ("988 Suicide & Crisis Lifeline", "Call or text 988 (24/7, free, confidential)"),
        ("Crisis Text Line", "Text HOME to 741741"),
        ("Emergency", "Call 911 if someone is in immediate danger"),
    ],
}

def get_secret(key: str) -> str | None:
    """Safe lookup: env var first, then st.secrets — without letting a
    missing secrets.toml file crash the app (st.secrets.get() raises
    StreamlitSecretNotFoundError when no secrets file exists at all,
    unlike a normal dict's .get())."""
    val = os.getenv(key)
    if val:
        return val
    try:
        return st.secrets.get(key)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# GUARDRAIL: crisis / safety detection (two-stage)
# ---------------------------------------------------------------------------
# Stage 1: regex — zero latency, zero external dependency, catches explicit
#          terms, and still works if Stage 2 is unavailable.
# Stage 2: OpenAI omni-moderation-latest — semantic pass that catches
#          paraphrases the regex can't enumerate ("I'm giving up",
#          "I can't do this anymore"). Gated on self-harm/intent and
#          self-harm/instructions specifically, NOT the general self-harm
#          category — this corpus is educational content ABOUT suicide and
#          self-harm, so the broad category alone would misfire constantly
#          on legitimate questions like "what are warning signs of suicide".
#
# Either stage tripping is treated as a crisis; the LLM never sees a message
# that trips this check, so it can't reason its way past the response.

CRISIS_PATTERNS = [
    r"\b(kill|end|hurt|harm)\s+(myself|my life)\b",
    r"\bsuicid(e|al)\b",
    r"\bwant(ing)?\s+to\s+die\b",
    r"\bdon'?t\s+want\s+to\s+(be alive|live|exist)\b",
    r"\bno\s+reason\s+to\s+live\b",
    r"\bself[\s-]?harm\b",
    r"\bcutting\s+myself\b",
    r"\bbetter\s+off\s+(dead|without me)\b",
    r"\b(overdose|od)\s+on\b",
    r"\bplan(ning)?\s+to\s+(kill|end)\b",
    r"\bgoodbye\s+forever\b",
    r"\b(giving up|give up)\s+(on\s+(life|everything|myself))?\b.*\b(can'?t|cannot)\b",
    r"\bcan'?t\s+(do this|go on|take (it|this)) anymore\b",
    r"\bno\s+point\s+(in\s+)?(living|anymore|trying)\b",
]
CRISIS_RE = re.compile("|".join(CRISIS_PATTERNS), re.IGNORECASE)

_moderation_warned = False


def _moderate_text(text: str) -> dict:
    """Call OpenAI's free moderation endpoint. Returns {} on any failure
    so the caller can fail safe rather than crash."""
    api_key = get_secret("OPENAI_API_KEY")
    if not api_key:
        return {}
    try:
        from openai import OpenAI
        client = OpenAI(api_key=api_key)
        result = client.moderations.create(model="omni-moderation-latest", input=text)
        cats = result.results[0].categories
        return {
            "self_harm_intent": getattr(cats, "self_harm_intent", False),
            "self_harm_instructions": getattr(cats, "self_harm_instructions", False),
        }
    except Exception:
        return {}


def detect_crisis(text: str) -> bool:
    global _moderation_warned

    # Stage 1: fast local regex
    if CRISIS_RE.search(text):
        return True

    # Stage 2: semantic moderation pass (only if the regex was clean)
    mod = _moderate_text(text)
    if not mod and not _moderation_warned:
        # No key configured, or the API call failed — surface it once so
        # you notice you're running on regex-only coverage.
        st.session_state.setdefault("_mod_unavailable", True)
        _moderation_warned = True
    return bool(mod.get("self_harm_intent") or mod.get("self_harm_instructions"))


def crisis_response(region: str = "US") -> str:
    lines = [
        "I'm really glad you reached out, and I want to make sure you get support "
        "beyond what I can offer here. I'm an educational chatbot, not a crisis service, "
        "but real help is available right now:",
        "",
    ]
    for name, detail in CRISIS_RESOURCES.get(region, CRISIS_RESOURCES["US"]):
        lines.append(f"- **{name}** — {detail}")
    lines.append("")
    lines.append(
        "If you're in immediate danger, please call emergency services or go to your "
        "nearest emergency room. You don't have to go through this alone."
    )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# RAG PIPELINE (cached so models/DB load once per session)
# ---------------------------------------------------------------------------

@st.cache_resource(show_spinner="Loading embedding model...")
def load_embedder():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(EMBED_MODEL)


@st.cache_resource(show_spinner="Connecting to vector store...")
def load_vectorstore():
    import chromadb
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    return client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )


@st.cache_resource(show_spinner=False)
def load_llm():
    from langchain_groq import ChatGroq
    api_key = get_secret("GROQ_API_KEY")
    if not api_key:
        st.error(
            "No GROQ_API_KEY found. Set it in a local .env file or in "
            "Streamlit Cloud's Secrets manager."
        )
        st.stop()
    return ChatGroq(groq_api_key=api_key, model_name=GROQ_MODEL, temperature=0.2, max_tokens=800)


def retrieve(query: str, embedder, collection, top_k: int = TOP_K):
    query_embedding = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[query_embedding], n_results=top_k)
    docs = []
    if results and results.get("documents"):
        for doc, meta, dist in zip(
            results["documents"][0], results["metadatas"][0], results["distances"][0]
        ):
            docs.append({"content": doc, "metadata": meta, "score": 1 - dist})
    return docs


SYSTEM_PROMPT = """You are Blissbot, a mental health EDUCATION assistant. Your role is strictly
educational and supportive — you are not a therapist, doctor, or crisis counselor.

Rules you must follow:
- Answer ONLY using the provided context. If the context doesn't cover the question, say so
  plainly and suggest the person consult a qualified professional — do not invent facts.
- Never diagnose, prescribe, or give clinical treatment advice.
- Never claim certainty about an individual's mental state; you only have what they've written.
- Keep a warm, non-judgmental, plain-language tone. Avoid clinical jargon unless you explain it.
- If the question touches on self-harm, suicide, or crisis, keep your answer supportive and brief,
  and defer to the crisis resources already shown to the user rather than repeating them yourself.
- Always gently encourage professional support for anything beyond general education.
"""


def generate_answer(query: str, context_docs: list, llm) -> str:
    if not context_docs:
        return (
            "I don't have information on that in my current knowledge base. "
            "For personalized guidance, please reach out to a licensed mental health "
            "professional, I'd rather not guess."
        )
    context = "\n\n".join(d["content"] for d in context_docs)
    prompt = f"""{SYSTEM_PROMPT}

Context:
{context}

Question: {query}

Answer (concise, plain language):"""
    response = llm.invoke(prompt)
    return response.content


# ---------------------------------------------------------------------------
# UI
# ---------------------------------------------------------------------------

st.set_page_config(page_title="Blissbot: Mental Health Education", page_icon="🌱", layout="wide")

with st.sidebar:
    st.markdown("### 🌱 About Blissbot")
    st.markdown(
        "Blissbot is an educational chatbot. It shares general information about mental "
        "health topics drawn from NIMH resources and curated Q&A datasets. **It is not a "
        "substitute for professional care, therapy, or crisis intervention.**"
    )
    st.divider()
    st.markdown("### 🚨 If you're in crisis, right now")
    for name, detail in CRISIS_RESOURCES["US"]:
        st.markdown(f"**{name}**  \n{detail}")
    st.divider()
    st.caption(
        "This chatbot does not store personal data beyond your current session and is "
        "not a HIPAA-covered service. Avoid sharing identifying personal details."
    )
    if st.session_state.get("_mod_unavailable"):
        st.warning(
            "Semantic safety check (OpenAI moderation) is not configured or unreachable — "
            "running on keyword-based crisis detection only. Set OPENAI_API_KEY to enable "
            "full coverage.",
            icon="⚠️",
        )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

st.title("🌱 Blissbot")
st.caption("A virtual mental health educator: information, not diagnosis.")

if "messages" not in st.session_state:
    st.session_state.messages = [
        {
            "role": "assistant",
            "content": (
                "Hi, I'm Blissbot. I can help explain mental health topics, coping "
                "concepts, and where to find support. What's on your mind?"
            ),
        }
    ]

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

user_input = st.chat_input("Ask about a mental health topic...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        placeholder = st.empty()

        # --- Guardrail check happens BEFORE any retrieval/LLM call ---
        if detect_crisis(user_input):
            answer = crisis_response()
            placeholder.markdown(answer)
        else:
            with st.spinner("Thinking..."):
                embedder = load_embedder()
                collection = load_vectorstore()
                llm = load_llm()
                docs = retrieve(user_input, embedder, collection)
                answer = generate_answer(user_input, docs, llm)
            placeholder.markdown(answer)

            with st.expander("Sources used"):
                if docs:
                    for d in docs:
                        meta = d["metadata"]
                        st.markdown(
                            f"- **{meta.get('source', 'unknown')}** "
                            f"(topic: {meta.get('topic_group', 'n/a')}, "
                            f"score: {d['score']:.2f}) — "
                            f"[link]({meta.get('url', '#')})"
                        )
                else:
                    st.markdown("No sources retrieved.")

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
st.caption(
    "Blissbot provides general educational information only and cannot replace "
    "professional diagnosis or treatment. In an emergency, call 911 (US) or your local "
    "emergency number."
)