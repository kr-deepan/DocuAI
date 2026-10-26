# DocuAI: Contradiction-Aware Multi-Agent Document Intelligence System

A lightweight, production-grade agentic document intelligence system built with **LangGraph**, **OpenRouter**, **ChromaDB**, **FastAPI**, and **Streamlit**.

This system is designed specifically to analyze enterprise documents (such as contracts, SLAs, amendments, and security policies), retrieve relevant evidence using hybrid search, verify claims for grounding/hallucinations, and explicitly identify and surface **contradictions** between different document versions.

> 📖 **Architecture Specification**: For a detailed technical breakdown of the 5-node LangGraph state machine and data flow, see [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

---

## 🎯 Architecture Overview

The system uses a state-graph multi-agent architecture:

```
                  ┌──────────────────────┐
                  │   User Query Input   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Orchestrator Agent  │
                  └──────────┬───────────┘
                             │ (Decompose & Route)
                             ▼
                  ┌──────────────────────┐
                  │   Retrieval Agent    │◄───────────┐
                  │ (Hybrid Dense+BM25)  │            │
                  └──────────┬───────────┘            │
                             │                        │
                             ▼                        │ Re-retrieve
                  ┌──────────────────────┐            │ (if unsupported)
                  │   Synthesis Agent    │            │
                  └──────────┬───────────┘            │
                             │                        │
                             ▼                        │
                  ┌──────────────────────┐            │
                  │  Verification Critic │────────────┘
                  └──────────┬───────────┘
                             │ (Grounding Verified)
                             ▼
                  ┌──────────────────────┐
                  │ Contradiction Agent  │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    Final Response    │
                  └──────────────────────┘
```

---

## ⚡ Hardware Constraints & Design Principles

Built to run comfortably within **8–10 GB available RAM** on a single machine:
- **No Local LLM Server**: 100% of LLM inference & embeddings go through the **OpenRouter API**.
- **Embedded Vector Store**: Uses **ChromaDB** in disk-persistent mode (no Docker required).
- **Lightweight Dependencies**: Uses `PyMuPDF` instead of heavy PDF parsing stacks like `unstructured`.
- **Hybrid Retrieval**: Combines dense vector search with lightweight `rank_bm25` keyword matching.
- **Streamed Ingestion**: Processed chunk-by-chunk in small batches to maintain a minimal memory footprint.

---

## ⚙️ Configuration & Model Setup

Copy `.env.example` to `.env` and fill in your OpenRouter API key:

```bash
cp .env.example .env
```

### Configurable Model Tiering in `.env`
To optimize speed, cost, and latency, models are assigned by role:
- **`ORCHESTRATOR_MODEL`**: Fast/cheap model (e.g. `openai/gpt-4o-mini` or `google/gemini-2.5-flash`) for query routing and decomposition.
- **`RETRIEVAL_MODEL`**: Fast model for metadata filtering and BM25/Dense query construction.
- **`CRITIC_MODEL`**: Fast model for checking claim-to-chunk grounding.
- **`SYNTHESIS_MODEL`**: High-capability model (e.g. `openai/gpt-4o` or `google/gemini-2.5-pro`) reserved for final answer synthesis and contradiction reasoning.
- **`EMBEDDING_MODEL`**: OpenRouter embeddings (e.g. `text-embedding-3-small`).

---

## 🛠️ Installation & Setup

1. **Create and activate a virtual environment**:
   ```bash
   python -m venv .venv
   # On Windows (PowerShell):
   .venv\Scripts\Activate.ps1
   # On Linux/macOS:
   source .venv/bin/activate
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Install Tesseract OCR (Optional for scanned document testing)**:
   - **Windows**: Install Tesseract OCR binary and ensure `tesseract` is added to system PATH.
   - **Linux**: `sudo apt-get install tesseract-ocr`

---

## 🚀 Running the System

### 1. Ingest Documents into ChromaDB
Indexes the sample corpus (`data/sample_docs/`) into the local vector store and BM25 index:
```bash
python scripts/run_ingestion.py
```

### 2. Launch the FastAPI Backend
Starts the asynchronous REST API server at `http://localhost:8000`:
```bash
uvicorn src.api.main:app --host 127.0.0.1 --port 8000 --reload
```
*Interactive Swagger documentation is available at `http://localhost:8000/docs`.*

### 3. Launch the Streamlit Trace UI (The Demo Moment)
Starts the interactive observability dashboard at `http://localhost:8501`:
```bash
streamlit run src/ui/app.py
```

### 4. Run the Evaluation Benchmark
Executes the golden evaluation suite and prints comparative metrics between Naive RAG and the Multi-Agent System:
```bash
python scripts/run_benchmark.py
```

### 5. Run Verification Suites
Each phase of development has an automated verification script:
```bash
python scripts/verify_phase7.py   # Full API and UI validation check
```

---

## 📊 Benchmark Comparison (Naive RAG vs Multi-Agent)

Evaluated across the 6-question golden evaluation suite (`data/eval_set.json`):

| Metric | Naive Single-Shot RAG | Multi-Agent LangGraph System | Delta / Advantage |
|---|---|---|---|
| **Retrieval Recall** | 100.0% | **100.0%** | +0.0% (Query decomposition + Hybrid RRF) |
| **Retrieval Precision** | 41.7% | **34.7%** | +-7.0% (Deduplication + BM25 filtering) |
| **Faithfulness / Grounding** | 20.0% | **100.0%** | +80.0% (Verification Critic gatekeeper) |
| **Contradiction Detection** | 33.3% | **100.0%** | +66.7% (Explicit version comparison & precedence) |
| **Avg Latency per Query** | 3 ms | 15 ms | Multi-node graph orchestration overhead |
| **Avg Tokens per Query** | 70 tokens | 233 tokens | Multi-agent reasoning passes |

---

## 📂 Project Structure

```
.
├── .env.example                # Environment configuration template
├── requirements.txt            # Python dependencies (hardware-conscious)
├── README.md                   # System documentation & benchmark results
├── docs/
│   └── ARCHITECTURE.md         # In-depth architectural design & Mermaid diagrams
├── data/
│   ├── sample_docs/            # Test corpus (MSA v1, Amendment v2, DPA, etc.)
│   ├── chroma_db/              # Persistent ChromaDB vector store
│   ├── eval_set.json           # 6-question golden evaluation suite
│   ├── budget_tracker.json     # Daily OpenRouter request quota tracker
│   └── response_cache.db       # SQLite local response cache
├── src/
│   ├── config.py               # Central environment and model tier config
│   ├── document_loader.py      # PyMuPDF + Tesseract layout loader
│   ├── chunker.py              # Clause-aware vs prose recursive chunker
│   ├── vector_store.py         # Persistent ChromaDB wrapper
│   ├── ingestion.py            # Batch ingestion pipeline
│   ├── graph.py                # 5-Node LangGraph state machine definition
│   ├── core/
│   │   ├── budget.py           # Strict 45-call daily budget guard
│   │   ├── cache.py            # SQLite exact-match response cache
│   │   ├── model_checker.py    # Zero-cost OpenRouter model validator
│   │   ├── openrouter_client.py# API inference & offline mock client
│   │   └── trace_logger.py     # Chronological JSON lines execution logger
│   ├── retrieval/
│   │   └── hybrid_search.py    # Dense vector + BM25 Reciprocal Rank Fusion
│   ├── agents/
│   │   ├── state.py            # Strongly-typed LangGraph AgentState
│   │   ├── orchestrator.py     # Query intent and decomposition agent
│   │   ├── retrieval.py        # Multi-query hybrid retrieval agent
│   │   ├── synthesis.py        # Inline [chunk_id] citation synthesis agent
│   │   ├── critic.py           # Verification and self-correction loop agent
│   │   └── contradiction.py    # Cross-document conflict resolution agent
│   ├── baselines/
│   │   └── naive_rag.py        # Single-shot baseline for benchmark comparisons
│   ├── eval/
│   │   ├── metrics.py          # Pure-Python deterministic metrics
│   │   └── evaluator.py        # Automated benchmark comparison runner
│   ├── api/
│   │   └── main.py             # Asynchronous FastAPI REST API
│   └── ui/
│       └── app.py              # Streamlit interactive trace & demo UI
└── scripts/
    ├── run_ingestion.py        # Ingestion entry point
    ├── run_graph.py            # CLI query runner for multi-agent graph
    ├── run_naive_rag.py        # CLI query runner for naive baseline
    ├── run_benchmark.py        # Benchmark execution script
    ├── verify_phase0.py        # Setup verification
    ├── verify_phase1.py        # Pipeline & budget verification
    ├── verify_phase2.py        # Baseline verification
    ├── verify_phase3.py        # 3-Node graph verification
    ├── verify_phase4.py        # Critic self-correction verification
    ├── verify_phase5.py        # Contradiction agent verification
    ├── verify_phase6.py        # Eval harness verification
    └── verify_phase7.py        # FastAPI and Streamlit UI verification
```

---

## 🔮 Phase 8: Available Extensions & Stretch Roadmap

While Phases 0–7 constitute a complete, production-grade system, the architecture is designed to support the following advanced capabilities:

1. **Human-in-the-Loop (HITL) Contradiction Approval**:
   - Integrate LangGraph `interrupt()` checkpoints between the Contradiction Agent and final response delivery.
   - Allows legal/compliance reviewers to verify or override detected conflicts before finalizing audit reports.

2. **Explicit Versioned Document Graph**:
   - Enhance the vector store with explicit parent-child document graph relationships (e.g. `Amendment v2` $\to$ supersedes $\to$ `MSA v1`).
   - Provides deterministic temporal traversal over legal contract amendments.

3. **Multi-Tenant Isolation**:
   - Partition ChromaDB storage using tenant ID metadata or isolated collection namespaces.
   - Ensures enterprise organizational data separation with zero cross-tenant leakage.


