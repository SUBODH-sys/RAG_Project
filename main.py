import os
import re
import time
import numpy as np
import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# CONFIG


PERSIST_DIR = "data/vector_store"        
COLLECTION_NAME = "json_qa_documents"
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
GROQ_MODEL = "qwen/qwen3.8-27b"               
TOP_K = 3

FUSION_POOL_SIZE = 20   # candidates pulled from EACH retriever before fusion
RRF_K = 10              # RRF damping constant

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


# GUARDRAIL: crisis / safety detection (two-stage)

# Stage 1: regex 
# Stage 2: OpenAI omni-moderation-latest semantic pass that catches paraphrases the regex can't enumerate.

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


# RAG PIPELINE (cached so models/DB/indexes load once per session)

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


def _tokenize(text: str) -> list[str]:
    """Simple, dependency-free tokenizer: lowercase, strip punctuation, split
    on whitespace. Must match the tokenizer used to build the notebook's
    BM25 index so scoring behavior stays consistent."""
    return re.findall(r"[a-z0-9]+", text.lower())


class BM25Index:
    """Builds and queries a BM25 index over the documents already stored in
    the Chroma collection. Built once per session (cached) from the same
    documents as the dense index, so it can never drift out of sync."""

    def __init__(self, collection):
        from rank_bm25 import BM25Okapi
        all_docs = collection.get(include=["documents"])
        self.doc_ids = all_docs["ids"]
        self.doc_texts = all_docs["documents"]
        tokenized_corpus = [_tokenize(doc) for doc in self.doc_texts]
        self.bm25 = BM25Okapi(tokenized_corpus)

    def query(self, query_text: str, top_k: int) -> list[dict]:
        """Returns top_k candidates ranked best-first as {'id','content','score'} dicts."""
        scores = self.bm25.get_scores(_tokenize(query_text))
        top_indices = np.argsort(scores)[::-1][:top_k]
        return [
            {"id": self.doc_ids[i], "content": self.doc_texts[i], "score": float(scores[i])}
            for i in top_indices
        ]


class HybridRetriever:
    """Combines dense (cosine-similarity) retrieval with BM25 lexical
    retrieval via Reciprocal Rank Fusion.

    retrieve() returns a list of dicts shaped like:
        {'id', 'content', 'metadata', 'score', 'distance', 'rank'}
    """

    def __init__(self, embedder, collection, bm25_index: BM25Index,
                 fusion_pool_size: int = FUSION_POOL_SIZE, rrf_k: int = RRF_K):
        self.embedder = embedder
        self.collection = collection
        self.bm25_index = bm25_index
        self.fusion_pool_size = fusion_pool_size
        self.rrf_k = rrf_k
        self.last_error = None

    def _dense_candidates(self, query: str, pool_size: int) -> dict:
        """Returns {doc_id: {'content','metadata','similarity_score','distance'}}."""
        query_embedding = self.embedder.encode([query])[0].tolist()
        results = self.collection.query(query_embeddings=[query_embedding], n_results=pool_size)

        candidates = {}
        if results.get("documents") and results["documents"][0]:
            for doc_id, document, metadata, distance in zip(
                results["ids"][0], results["documents"][0],
                results["metadatas"][0], results["distances"][0],
            ):
                candidates[doc_id] = {
                    "content": document,
                    "metadata": metadata,
                    "similarity_score": 1 - distance,
                    "distance": distance,
                }
        return candidates

    def retrieve(self, query: str, top_k: int = TOP_K, score_threshold: float = 0.3) -> list[dict]:
        """Retrieve relevant documents for a query using fused dense + BM25 ranking."""
        self.last_error = None
        try:
            dense_candidates = self._dense_candidates(query, self.fusion_pool_size)
            bm25_candidates = self.bm25_index.query(query, self.fusion_pool_size)

            rrf_scores: dict = {}
            for rank, doc_id in enumerate(dense_candidates.keys(), start=1):
                rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (self.rrf_k + rank)
            for rank, cand in enumerate(bm25_candidates, start=1):
                rrf_scores[cand["id"]] = rrf_scores.get(cand["id"], 0.0) + 1.0 / (self.rrf_k + rank)

            fused_order = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
            bm25_lookup = {c["id"]: c for c in bm25_candidates}

            retrieved_docs = []
            for i, (doc_id, fused_score) in enumerate(fused_order):
                if fused_score < score_threshold:
                    continue

                if doc_id in dense_candidates:
                    content = dense_candidates[doc_id]["content"]
                    metadata = dense_candidates[doc_id]["metadata"]
                    distance = dense_candidates[doc_id]["distance"]
                else:
                    content = bm25_lookup[doc_id]["content"]
                    metadata = {}
                    distance = None

                retrieved_docs.append({
                    "id": doc_id,
                    "content": content,
                    "metadata": metadata,
                    "score": fused_score,
                    "distance": distance,
                    "rank": i + 1,
                })

                if len(retrieved_docs) >= top_k:
                    break

            return retrieved_docs

        except Exception as e:
            import traceback
            self.last_error = f"{type(e).__name__}: {e}"
            print(f"Error during hybrid retrieval: {e}")
            print(traceback.format_exc())
            return []


@st.cache_resource(show_spinner="Building BM25 index...")
def load_bm25_index(_collection):
    return BM25Index(_collection)


@st.cache_resource(show_spinner=False)
def load_hybrid_retriever(_embedder, _collection, _bm25_index):
    return HybridRetriever(_embedder, _collection, _bm25_index)


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


def retrieve(query: str, retriever: HybridRetriever, top_k: int = TOP_K):
    return retriever.retrieve(query, top_k=top_k)


SYSTEM_PROMPT = """You are Blissbot, a mental health EDUCATION assistant. Your role is strictly
educational and supportive you are not a therapist, doctor, or crisis counselor.

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



# Streamlit UI


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
                bm25_index = load_bm25_index(collection)
                hybrid_retriever = load_hybrid_retriever(embedder, collection, bm25_index)
                llm = load_llm()
                docs = retrieve(user_input, hybrid_retriever)
                answer = generate_answer(user_input, docs, llm)
            placeholder.markdown(answer)

            with st.expander("Sources used"):
                if docs:
                    for d in docs:
                        meta = d["metadata"] or {}
                        st.markdown(
                            f"- **{meta.get('source', 'unknown')}** "
                            f"(topic: {meta.get('topic_group', 'n/a')}, "
                            f"fused score: {d['score']:.4f}) — "
                            f"[link]({meta.get('url', '#')})"
                        )
                elif hybrid_retriever.last_error:
                    st.error(f"Retrieval failed: {hybrid_retriever.last_error}")
                    st.caption("Full traceback printed to the terminal running `streamlit run`.")
                else:
                    st.markdown("No sources retrieved (retriever ran cleanly but found no matches).")

    st.session_state.messages.append({"role": "assistant", "content": answer})

st.divider()
st.caption(
    "Blissbot provides general educational information only and cannot replace "
    "professional diagnosis or treatment. In an emergency, call 911 (US) or your local "
    "emergency number."
)