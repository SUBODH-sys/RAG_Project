# 🌱 Blissbot: Mental Health Education RAG Chatbot

Blissbot is a Retrieval-Augmented Generation (RAG) chatbot that answers mental-health
education questions using a curated Q&A corpus (largely sourced from NIMH). It combines
dense and lexical retrieval, a Groq-hosted LLM for generation, and a two-stage guardrail
that detects crisis language and responds with hotline resources instead of a
generated answer.

> ⚠️ **Not a clinical tool.** Blissbot is strictly educational. It does not diagnose,
> treat, or provide crisis counseling. See [Safety & Guardrails](#safety--guardrails)
> below.

## Features

- **Hybrid retrieval** :— combines dense vector search (cosine similarity over
  sentence embeddings) with BM25 lexical search, fused via Reciprocal Rank Fusion (RRF).
- **Vector store** :— [ChromaDB](https://www.trychroma.com/) persistent collection,
  embedded with `BAAI/bge-small-en-v1.5` via `sentence-transformers`.
- **LLM generation** :— [Groq](https://groq.com/)-hosted model (`langchain-groq`),
  called only on documents retrieved from the corpus (answers are grounded in context,
  not free generation).
- **Crisis-detection guardrail** :— a regex pass plus an optional OpenAI moderation
  API pass catch self-harm / suicide language *before* any query reaches the LLM,
  and reply with region-specific crisis resources instead.
- **Streamlit chat UI** :— session-based chat interface with a sources panel showing
  which documents backed each answer and their fused retrieval score.
- **Evaluation notebooks** :— Jupyter notebooks comparing dense-only retrieval,
  hybrid retrieval, and end-to-end pipeline behavior.

## Project structure

```
RAG_Project/
├── main.py                  # Streamlit app hybrid (dense + BM25) retriever, current entry point                 
├── requirements.txt
├── data/
│   ├── json_files/
│   │   └── bliss_corpus.json    # ~2,964 Q&A documents across 14 topic groups
│   │                             # (e.g. depression, anxiety, trauma, suicide/self-harm,
│   │                             #  substance use, eating disorders, coping, general literacy)
│   ├── text_files/               # Sample plain-text documents (e.g. machine_learning.txt)
│   └── vector_store/             # Persisted ChromaDB collection (chroma.sqlite3 + index)
└── notebook/
    ├── RAG_Pipeline_Hybrid.ipynb
    ├── hybrid_search_evaluation.ipynb
    └── retriever_evaluation_dense.ipynb
```

## How it works

1. **Ingestion** :— Q&A pairs from `data/json_files/bliss_corpus.json` are embedded with
   `BAAI/bge-small-en-v1.5` and stored in a persistent ChromaDB collection
   (`data/vector_store/`), alongside metadata (`source`, `topic_group`, `url`, `region`, `flags`).
2. **Query time**:
   - **Guardrail check** :— the incoming message is checked against a regex pattern set
     for crisis language, then (if configured) an OpenAI moderation call. If either
     trips, the app returns crisis resources directly and skips retrieval/generation.
   - **Retrieval** :— the query is embedded and searched against Chroma (dense), and
     separately scored with a BM25 index built over the same corpus (lexical). Both
     ranked candidate lists are fused with Reciprocal Rank Fusion, and low-scoring
     fused results are filtered out.
   - **Generation** :— the top fused documents are passed as context to the Groq LLM
     under a system prompt that restricts it to answering from context only, forbids
     diagnosis/clinical advice, and asks for a warm, plain-language tone.
3. **UI** :— Streamlit renders the conversation, plus an expandable "Sources used"
   panel per answer showing which documents were retrieved and their score.

## Prerequisites

- Python 3.9+
- A [Groq API key](https://console.groq.com/) (used for answer generation)
- An [OpenAI API key](https://platform.openai.com/) (enables the semantic
  moderation pass for crisis detection; without it, the app falls back to regex-only
  detection and shows a one-time warning)

## Setup

```bash
# Clone the repo
git clone https://github.com/SUBODH-sys/RAG_Project.git
cd RAG_Project

# Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

Create a `.env` file in the project root with your keys:

```
GROQ_API_KEY=gsk_...
OPENAI_API_KEY=sk-...   #enables semantic crisis-moderation pass
```

(When deploying to Streamlit Community Cloud, set these in **App settings → Secrets**
instead of a `.env` file.)

## Running the app

The hybrid-retrieval version is the current entry point:

```bash
streamlit run main.py
```

The earlier dense-only version is also available:

```bash
streamlit run app.py
```

The vector store under `data/vector_store/` is pre-built from `bliss_corpus.json`, so
the app can run out of the box. To rebuild it from scratch or index new documents, see
the ingestion steps in `notebook/RAG_Pipeline_Hybrid.ipynb`.

## Notebooks

| Notebook | Purpose |
|---|---|
| `RAG_Pipeline_Hybrid.ipynb` | End-to-end pipeline: ingestion, embedding, hybrid retrieval, generation |
| `hybrid_search_evaluation.ipynb` | Evaluates the fused dense + BM25 retriever |
| `retriever_evaluation_dense.ipynb` | Evaluates the dense-only retriever baseline |

## Safety & guardrails

- Crisis-related messages (suicide, self-harm, etc.) are intercepted **before**
  retrieval or generation and answered with hotline resources (988 Suicide & Crisis
  Lifeline, Crisis Text Line, emergency services), not with a generated response.
- The LLM system prompt restricts it to answering only from retrieved context, bars
  diagnosis or treatment claims, and defers anything beyond general education to a
  licensed professional.
- The app does not persist chat history beyond the active session and is **not** a
  HIPAA-covered service avoid sharing identifying personal information with it.

## Tech stack

`langchain` · `langchain-community` · `langchain-groq` · `sentence-transformers` ·
`chromadb` · `rank_bm25` · `streamlit` · `numpy` · `pandas` · `matplotlib` ·
`python-dotenv`

## License

No license file is currently included in this repository. Add one (e.g. MIT,
Apache-2.0) if you intend for others to reuse this code.