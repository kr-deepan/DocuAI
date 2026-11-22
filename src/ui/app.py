"""
DocuAI: Contradiction-Aware Multi-Agent Document Intelligence Dashboard
Streamlit UI module (identical to root app.py for dual-path compatibility).
Supports isolated multi-project workspaces, document upload & extraction, SQLite persistence, and 5-node auditing.
"""
import sys
import os
from pathlib import Path

# Ensure backend root directory is at the front of sys.path
BASE_DIR = Path(__file__).resolve().parent.parent.parent
if str(BASE_DIR) not in sys.path:
    sys.path.insert(0, str(BASE_DIR))

import streamlit as st

# -----------------------------------------------------------------------------
# Page Configuration (Must be the very first Streamlit command)
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="DocuAI: Multi-Agent Document Intelligence",
    page_icon="⚖️",
    layout="wide",
    initial_sidebar_state="expanded"
)

from src.config import DAILY_REQUEST_BUDGET
from src.core.budget import budget_guard
from src.core.database import db_manager
from src.vector_store import vector_store
from src.ingestion import IngestionPipeline, ingest_project_file_bytes
from src.retrieval.hybrid_search import get_hybrid_retriever
from src.graph import run_agent_system
from src.baselines.naive_rag import naive_rag
from src.agents.synthesis import extract_citations

# -----------------------------------------------------------------------------
# Sidebar Configuration
# -----------------------------------------------------------------------------
st.sidebar.title("⚖️ DocuAI Workspace")

# 1. Mode Selection
execution_mode = st.sidebar.radio(
    "Execution Mode",
    ["Contradiction-Aware Multi-Agent (5 Nodes)", "Naive Single-Shot RAG (Baseline)"],
    help="Compare the 5-node multi-agent state graph against standard linear RAG"
)

st.sidebar.markdown("---")

# 2. Project Workspace Selector
st.sidebar.subheader("📁 Project Workspace")

projects = db_manager.list_projects()
project_map = {p["id"]: p for p in projects}
project_ids = list(project_map.keys())

if "selected_project_id" not in st.session_state or st.session_state["selected_project_id"] not in project_map:
    st.session_state["selected_project_id"] = "default"

def format_project_label(pid: str) -> str:
    p = project_map.get(pid, {})
    return f"{p.get('name', pid)} [{pid}] ({p.get('document_count', 0)} docs)"

selected_index = project_ids.index(st.session_state["selected_project_id"]) if st.session_state["selected_project_id"] in project_ids else 0

selected_project_id = st.sidebar.selectbox(
    "Active Workspace Project",
    options=project_ids,
    index=selected_index,
    format_func=format_project_label,
    help="All document uploads, ChromaDB embeddings, BM25 indices, and contradiction queries are strictly isolated to this project."
)
st.session_state["selected_project_id"] = selected_project_id
current_project = project_map.get(selected_project_id, {})

# Display active project metadata badge
st.sidebar.caption(f"📝 {current_project.get('description', 'Isolated workspace')}")

# 3. Create New Project Expander
with st.sidebar.expander("➕ Create New Project", expanded=False):
    new_proj_name = st.text_input("Project Name", placeholder="e.g. Healthcare Vendor Auditing", key="new_proj_name_input")
    new_proj_id = st.text_input("Project ID (slug)", placeholder="e.g. healthcare_vendor", key="new_proj_id_input")
    new_proj_desc = st.text_area("Description (optional)", placeholder="Contract review and SLA compliance auditing...", key="new_proj_desc_input")
    
    if st.button("Create Project", type="secondary", use_container_width=True):
        if new_proj_name.strip() and new_proj_id.strip():
            clean_id = new_proj_id.strip().lower().replace(" ", "_")
            if clean_id in project_map:
                st.sidebar.error(f"Project ID '{clean_id}' already exists!")
            else:
                try:
                    db_manager.create_project(clean_id, new_proj_name.strip(), new_proj_desc.strip())
                    st.session_state["selected_project_id"] = clean_id
                    st.sidebar.success(f"Workspace '{new_proj_name}' created!")
                    st.rerun()
                except Exception as e:
                    st.sidebar.error(f"Creation failed: {e}")
        else:
            st.sidebar.warning("Please provide both Name and ID.")

# 4. Upload Documents Expander
with st.sidebar.expander("📤 Upload Documents", expanded=False):
    st.write(f"Target Project: **{current_project.get('name', selected_project_id)}**")
    uploaded_files = st.file_uploader(
        "Upload files (.pdf, .txt, .png, .jpg)",
        type=["txt", "pdf", "png", "jpg", "jpeg"],
        accept_multiple_files=True,
        key=f"uploader_{selected_project_id}"
    )
    
    if uploaded_files and st.button("🚀 Ingest & Index", type="primary", use_container_width=True):
        total_indexed_chunks = 0
        progress_bar = st.progress(0.0)
        status_text = st.empty()
        
        for i, up_file in enumerate(uploaded_files):
            status_text.text(f"Processing '{up_file.name}'...")
            try:
                count = ingest_project_file_bytes(
                    project_id=selected_project_id,
                    file_bytes=up_file.getvalue(),
                    filename=up_file.name
                )
                total_indexed_chunks += count
            except Exception as e:
                st.error(f"Failed to ingest {up_file.name}: {e}")
            progress_bar.progress((i + 1) / len(uploaded_files))
            
        status_text.text("Ingestion and indexing complete!")
        st.success(f"Successfully indexed {len(uploaded_files)} document(s) ({total_indexed_chunks} chunks) into '{selected_project_id}'!")
        st.rerun()

# 5. Project Documents List & Deletion
proj_chunks = vector_store.get_count(selected_project_id)
proj_docs = db_manager.get_project_documents(selected_project_id)

st.sidebar.write(f"**Total Indexed Chunks**: `{proj_chunks}`")

with st.sidebar.expander(f"📑 Ingested Documents ({len(proj_docs)})", expanded=False):
    if proj_docs:
        for d in proj_docs:
            st.markdown(f"- **{d['filename']}**  \n  `{d.get('chunk_count', 0)} chunks` • `{d.get('file_type', 'txt')}` • `{d.get('file_size', 0):,} bytes`")
    else:
        st.info("No documents in this project yet. Use 'Upload Documents' above to add files.")

# Re-index button for active project
col_act1, col_act2 = st.sidebar.columns(2)
if col_act1.button("🔄 Re-Index", help="Re-sync hybrid retriever index for this project"):
    with st.spinner("Re-syncing BM25 index..."):
        get_hybrid_retriever(selected_project_id).refresh_index()
        st.sidebar.success("BM25 Index refreshed!")
        st.rerun()

if selected_project_id != "default":
    if col_act2.button("🗑️ Delete", help="Delete this project workspace and all its data"):
        vector_store.delete_project_collection(selected_project_id)
        db_manager.delete_project(selected_project_id)
        st.session_state["selected_project_id"] = "default"
        st.sidebar.success(f"Project '{selected_project_id}' deleted.")
        st.rerun()

st.sidebar.markdown("---")

# 6. Daily Budget Meter
st.sidebar.subheader("📊 OpenRouter Request Budget")
requests_used = budget_guard.get_usage()
ceiling = DAILY_REQUEST_BUDGET
budget_ratio = min(1.0, requests_used / max(1, ceiling))

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

# -----------------------------------------------------------------------------
# Main Header & Project Context Banner
# -----------------------------------------------------------------------------
st.title("⚖️ DocuAI: Multi-Agent Document Intelligence")
st.caption("Contradiction-Aware State-Graph Multi-Agent Architecture with Isolated Multi-Project Workspaces")

st.info(
    f"🔒 **Active Isolated Workspace**: `{current_project.get('name', selected_project_id)}` "
    f"(`ID: {selected_project_id}`) — **{proj_chunks} chunks** indexed across **{len(proj_docs)} documents**. "
    f"All retrieval, verification, and contradiction auditing strictly operates within this project."
)

# Sample Query Presets (Active for default legal corpus, or inspiration for custom projects)
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
    with st.spinner(f"Executing agent pipeline on workspace '{selected_project_id}'..."):
        if execution_mode.startswith("Naive"):
            # ---------------------------------------------------------
            # Naive Baseline Execution (Project-Scoped)
            # ---------------------------------------------------------
            res = naive_rag.query(query, project_id=selected_project_id)
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

            st.markdown(f"### 🔍 Retrieved Chunks ({len(retrieved)} from workspace '{selected_project_id}')")
            for idx, c in enumerate(retrieved, 1):
                meta = c.get("metadata", {})
                with st.expander(f"Chunk {idx}: {meta.get('source_doc')} - {meta.get('section_header')}"):
                    st.write(c.get("text", ""))

        else:
            # ---------------------------------------------------------
            # Full 5-Node Multi-Agent Execution (Project-Scoped)
            # ---------------------------------------------------------
            final_state = run_agent_system(query, project_id=selected_project_id)
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
            # Visual Step-by-Step Agent Trace
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
                st.write(f"**Role**: Executes Dense Vector + BM25 keyword search strictly against workspace `{selected_project_id}`.")
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
                st.write("**Role**: Compares claims across document versions in this workspace and resolves governing precedence.")
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
            st.markdown(f"### 📄 Final Grounded Answer & Contradiction Alerts (Workspace: `{selected_project_id}`)")
            st.markdown(final_ans)

            st.markdown("---")

            # ---------------------------------------------------------
            # Evidence Chunks Accordion Viewer
            # ---------------------------------------------------------
            st.markdown(f"### 📚 Retrieved Evidence Documents ({len(retrieved)} chunks)")
            for idx, c in enumerate(retrieved, 1):
                meta = c.get("metadata", {})
                cid = c.get("chunk_id")
                is_cited = "✅ Cited" if cid in citations else "Uncited"
                with st.expander(f"Chunk {idx}: [{cid}] {meta.get('source_doc')} - {meta.get('section_header')} ({is_cited})"):
                    st.write(f"**RRF Score**: `{c.get('rrf_score', 0):.4f}` | **Version**: `{meta.get('version')}` | **Project**: `{meta.get('project_id', selected_project_id)}`")
                    st.markdown(f"```text\n{c.get('text', '')}\n```")
