"""
Streamlit Interactive Dashboard (Phase 7 - The Demo Moment)

Role in the System:
-------------------
Provides a visual, interactive demonstration dashboard for the Contradiction-Aware
Multi-Agent Document Intelligence System.

Key Features:
-------------
1. Visual Step-by-Step Agent Trace: Exposes internal state transitions of
   Orchestrator -> Retrieval -> Synthesis -> Critic -> Contradiction Analyzer.
2. Side-by-Side Mode Toggle: Allows instantaneous comparison between
   Multi-Agent System and the Naive Single-Shot RAG baseline.
3. Live Budget Quota Gauge: Real-time tracking of daily OpenRouter request usage.
4. 1-Click Sample Query Presets: Demonstrates notice conflicts, payment conflicts,
   and policy overrides with a single click.
"""
import sys
import os
import streamlit as st

# Add backend directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from src.config import DAILY_REQUEST_BUDGET
from src.core.budget import budget_guard
from src.vector_store import vector_store
from src.graph import run_agent_system
from src.baselines.naive_rag import naive_rag
from src.agents.synthesis import extract_citations

# -----------------------------------------------------------------------------
# Page Configuration & Styling
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Contradiction-Aware Multi-Agent Document Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom sleek styling
st.markdown("""
<style>
    .main-title {
        font-size: 2.2rem;
        font-weight: 700;
        color: #1E88E5;
        margin-bottom: 0.2rem;
    }
    .sub-title {
        font-size: 1.05rem;
        color: #78909C;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #F8F9FA;
        border-radius: 8px;
        padding: 12px;
        border-left: 4px solid #1E88E5;
        margin-bottom: 10px;
    }
    .trace-node {
        background-color: #F0F4F8;
        border-radius: 8px;
        padding: 14px;
        margin-bottom: 12px;
        border-left: 5px solid #00ACC1;
    }
    .conflict-box {
        background-color: #FFF3E0;
        border-radius: 8px;
        padding: 16px;
        border-left: 5px solid #FF9800;
        margin-top: 15px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.title("⚙️ System Control")

# Mode Selection
execution_mode = st.sidebar.radio(
    "Execution Mode",
    ["Contradiction-Aware Multi-Agent (5 Nodes)", "Naive Single-Shot RAG (Baseline)"],
    help="Compare the multi-agent state graph against standard linear RAG"
)

st.sidebar.markdown("---")

# Daily Budget Meter
st.sidebar.subheader("📊 OpenRouter Request Budget")
requests_used = budget_guard.get_usage()
ceiling = DAILY_REQUEST_BUDGET
budget_ratio = min(1.0, requests_used / ceiling)

st.sidebar.progress(budget_ratio)
col_b1, col_b2 = st.sidebar.columns(2)
col_b1.metric("Used Today", f"{requests_used} / {ceiling}")
col_b2.metric("Remaining", f"{max(0, ceiling - requests_used)}")

if requests_used >= ceiling:
    st.sidebar.error("⚠️ Daily Request Budget Reached! Next calls will be halted to preserve headroom.")
elif requests_used >= 35:
    st.sidebar.warning("⚡ Approaching daily budget ceiling.")
else:
    st.sidebar.success("✅ Quota within safe operational limits.")

st.sidebar.markdown("---")

# Document Corpus Explorer
st.sidebar.subheader("📂 Document Corpus")
stored = vector_store.collection.get()
total_chunks = len(stored.get("ids", [])) if stored else 0
st.sidebar.write(f"**Total Indexed Chunks**: `{total_chunks}`")

with st.sidebar.expander("View Ingested Documents"):
    if stored and stored.get("metadatas"):
        docs_summary = {}
        for m in stored["metadatas"]:
            doc = m.get("source_doc", "unknown")
            docs_summary[doc] = docs_summary.get(doc, 0) + 1
        for d, count in docs_summary.items():
            st.write(f"- `{d}` ({count} chunks)")
    else:
        st.write("No documents indexed.")

# -----------------------------------------------------------------------------
# Header
# -----------------------------------------------------------------------------
st.markdown("<div class='main-title'>Contradiction-Aware Multi-Agent Document Intelligence</div>", unsafe_allow_html=True)
st.markdown("<div class='sub-title'>State-Graph Multi-Agent Architecture for Enterprise Contract & Policy Auditing</div>", unsafe_allow_html=True)

# Sample Query Presets
st.markdown("##### 💡 1-Click Sample Presets:")
preset_cols = st.columns(4)

selected_preset = None
if preset_cols[0].button("⚖️ Notice Period (30 vs 90d)"):
    selected_preset = "What is the termination notice period required under the contract and its amendment?"
if preset_cols[1].button("💳 Payment Terms (Net-30 vs 60)"):
    selected_preset = "What are the invoicing payment terms and late fees across the agreement versions?"
if preset_cols[2].button("⏱️ Breach Notice (72h vs 24h)"):
    selected_preset = "Within how many hours must a data security breach incident be reported?"
if preset_cols[3].button("☁️ Scope of Services (Factual)"):
    selected_preset = "What specific cloud infrastructure and software services does Provider agree to deliver?"

# -----------------------------------------------------------------------------
# Main Query Input
# -----------------------------------------------------------------------------
default_query = selected_preset or "Compare the termination notice period and payment terms between the contract and amendment."
query = st.text_input("Enter Document Query:", value=default_query)

run_button = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

# -----------------------------------------------------------------------------
# Execution & Trace Display
# -----------------------------------------------------------------------------
if run_button and query.strip():
    with st.spinner("Executing agent pipeline..."):
        if execution_mode.startswith("Naive"):
            # ---------------------------------------------------------
            # Naive Baseline Execution
            # ---------------------------------------------------------
            res = naive_rag.query(query)
            answer = res.get("answer", "")
            retrieved = res.get("retrieved_chunks", [])
            latency = res.get("latency_ms", 0)
            tokens = res.get("usage", {}).get("total_tokens", 0)

            st.info("ℹ️ Running **Naive Single-Shot RAG Baseline** (Zero agent routing, zero verification, zero contradiction detection).")

            col_m1, col_m2, col_m3 = st.columns(3)
            col_m1.metric("End-to-End Latency", f"{latency} ms")
            col_m2.metric("Tokens Consumed", f"{tokens}")
            col_m3.metric("Contradiction Alert", "None (Blind)")

            st.markdown("### 📝 Synthesized Answer")
            st.markdown(answer)

            st.markdown("### 🔍 Retrieved Chunks (Stuffed Directly into Prompt)")
            for idx, c in enumerate(retrieved, 1):
                meta = c.get("metadata", {})
                with st.expander(f"Chunk {idx}: {meta.get('source_doc')} - {meta.get('section_header')}"):
                    st.write(c.get("text", ""))

        else:
            # ---------------------------------------------------------
            # Full 5-Node Multi-Agent Execution
            # ---------------------------------------------------------
            final_state = run_agent_system(query)
            traces = final_state.get("trace_logs", [])
            retrieved = final_state.get("retrieved_chunks", [])
            citations = final_state.get("citations", [])
            conflicts = final_state.get("contradictions_found", [])
            final_ans = final_state.get("final_answer", "")

            # Performance Metrics Bar
            total_tokens = sum([t.get("tokens", 0) for t in traces])
            total_latency = sum([t.get("latency_ms", 0) for t in traces])
            
            m_col1, m_col2, m_col3, m_col4 = st.columns(4)
            m_col1.metric("Nodes Executed", f"{len(traces)}")
            m_col2.metric("Total Latency", f"{total_latency} ms")
            m_col3.metric("Tokens Consumed", f"{total_tokens}")
            m_col4.metric("Contradictions Found", f"{len(conflicts)}")

            st.markdown("---")

            # ---------------------------------------------------------
            # THE DEMO MOMENT: Visual Step-by-Step Agent Trace
            # ---------------------------------------------------------
            st.markdown("### 🧠 Live Agent State Machine Trace")
            trace_tabs = st.tabs([
                "1. 🧠 Orchestrator",
                "2. 🔍 Hybrid Retriever",
                "3. ✍️ Synthesis",
                "4. 🛡️ Verification Critic",
                "5. ⚖️ Contradiction Agent"
            ])

            # Tab 1: Orchestrator
            with trace_tabs[0]:
                orch_trace = next((t for t in traces if t.get("agent_name") == "orchestrator"), {})
                st.write("**Role**: Analyzes query intent and decomposes into atomic search sub-queries.")
                st.write(f"- **Execution Time**: `{orch_trace.get('latency_ms', 0)} ms` | **Tokens**: `{orch_trace.get('tokens', 0)}`")
                sub_q = final_state.get("sub_queries", [])
                st.write("**Decomposed Sub-Queries**:")
                for sq in sub_q:
                    st.info(f"🔎 `{sq}`")

            # Tab 2: Retrieval
            with trace_tabs[1]:
                ret_trace = next((t for t in traces if t.get("agent_name") == "retriever"), {})
                st.write("**Role**: Executes Dense Vector + BM25 keyword search merged via Reciprocal Rank Fusion (RRF).")
                st.write(f"- **Retrieved Chunks Count**: `{len(retrieved)}` | **Strategy**: `Hybrid Dense + BM25 (RRF k=60)`")
                for idx, c in enumerate(retrieved, 1):
                    meta = c.get("metadata", {})
                    st.write(f"**[{idx}] {c.get('chunk_id')}** — `{meta.get('source_doc')}` | Section: `{meta.get('section_header')}` | RRF: `{c.get('rrf_score', 0):.4f}`")

            # Tab 3: Synthesis
            with trace_tabs[2]:
                synth_trace = next((t for t in traces if t.get("agent_name") == "synthesis"), {})
                st.write("**Role**: Synthesizes grounded draft answer with inline chunk ID citations.")
                st.write(f"- **Execution Time**: `{synth_trace.get('latency_ms', 0)} ms` | **Tokens**: `{synth_trace.get('tokens', 0)}`")
                st.write(f"**Extracted Chunk Citations ({len(citations)})**:")
                st.code(citations)

            # Tab 4: Critic
            with trace_tabs[3]:
                critic_trace = next((t for t in traces if t.get("agent_name") == "critic"), {})
                st.write("**Role**: Verifies that every claim is grounded and checks for hallucinated citations.")
                v_passed = final_state.get("verification_passed", True)
                attempts = final_state.get("verification_attempts", 1)
                
                if v_passed:
                    st.success(f"✅ Grounding Verification Passed on Attempt {attempts}!")
                else:
                    st.error(f"❌ Verification Failed on Attempt {attempts} - Triggering Re-retrieval loop!")
                st.write(f"**Critic Feedback**: {final_state.get('critic_feedback')}")

            # Tab 5: Contradiction Agent
            with trace_tabs[4]:
                contra_trace = next((t for t in traces if t.get("agent_name") == "contradiction"), {})
                st.write("**Role**: Compares claims across document versions and resolves governing precedence.")
                if conflicts:
                    st.warning(f"⚠️ Detected {len(conflicts)} Cross-Document Contradiction(s)!")
                    for idx, conf in enumerate(conflicts, 1):
                        st.write(f"**Conflict {idx}: {conf.get('dimension')}**")
                        st.write(f"- Earlier Term: {conf.get('v1_claim')}")
                        st.write(f"- Amended Term: {conf.get('v2_claim')}")
                        st.write(f"- Governing Authority: `{conf.get('governing_document')}`")
                        st.write(f"- Rationale: {conf.get('resolution_rationale')}")
                else:
                    st.success("✅ No cross-document contradictions detected.")

            st.markdown("---")

            # ---------------------------------------------------------
            # Final Verified Answer Display
            # ---------------------------------------------------------
            st.markdown("### 📄 Final Grounded Answer & Contradiction Alerts")
            st.markdown(final_ans)

            st.markdown("---")

            # ---------------------------------------------------------
            # Evidence Chunks Accordion Viewer
            # ---------------------------------------------------------
            st.markdown("### 📚 Retrieved Evidence Documents")
            for idx, c in enumerate(retrieved, 1):
                meta = c.get("metadata", {})
                cid = c.get("chunk_id")
                is_cited = "✅ Cited" if cid in citations else "Uncited"
                with st.expander(f"Chunk {idx}: [{cid}] {meta.get('source_doc')} - {meta.get('section_header')} ({is_cited})"):
                    st.write(f"**RRF Score**: `{c.get('rrf_score', 0):.4f}` | **Version**: `{meta.get('version')}`")
                    st.markdown(f"```text\n{c.get('text', '')}\n```")
