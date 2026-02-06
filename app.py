import os
import time
import tempfile
import streamlit as st
import streamlit.components.v1 as components

from insightgraph_pipeline import generate_knowledge_graph, PipelineConfig

# ---------------- Streamlit Page Config ----------------
st.set_page_config(
    page_title="Data2Dash – Knowledge Graph Extractor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------------- Custom CSS ----------------
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; color: #212529; }
    .stButton>button {
        width: 100%; border-radius: 8px; height: 3.5em;
        background-color: #4b6cb7; color: white; font-weight: 600; border: none;
        transition: all 0.3s ease;
    }
    .stButton>button:hover {
        background-color: #3a539b; box-shadow: 0 4px 12px rgba(75, 108, 183, 0.3); transform: translateY(-1px);
    }
    .stSidebar { background-color: #111111; color: #ffffff; }
    .stSidebar [data-testid="stMarkdownContainer"] p { color: #ffffff !important; }
    .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar label { color: #ffffff !important; }
    .stSidebar .stSelectbox label, .stSidebar .stRadio label, .stSidebar .stCheckbox label { color: #ffffff !important; }
    h1 { color: #1a1c23; font-family: 'Inter', sans-serif; font-weight: 700; }
    [data-testid="stAppViewContainer"] .main .block-container { padding-top: 1.5rem; }
    .data2dash-brand { font-size: 0.85rem; color: #6c757d; margin-top: -0.5rem; }
    .stMarkdown { color: #495057; }
    </style>
""", unsafe_allow_html=True)

# ---------------- Main Title ----------------
st.title("📊 Data2Dash")
st.markdown("**Knowledge Graph Extractor** — Transform research papers and text into interactive, relational knowledge graphs.")
st.caption("Data2Dash project")

# ---------------- Sidebar ----------------
st.sidebar.image("https://img.icons8.com/wired/128/ffffff/bar-chart.png", width=80)
st.sidebar.title("Data2Dash")
st.sidebar.caption("Configuration")

input_method = st.sidebar.selectbox(
    "Select Input Source:",
    ["📄 Upload PDF/TXT", "✍️ Manual Text Input"]
)

st.sidebar.divider()

# ---------- Performance / Quality knobs ----------
with st.sidebar.expander("⚙️ Extraction Settings", expanded=False):
    max_chunks = st.slider("Max chunks (cost control)", 10, 60, 40, 2)
    top_k = st.slider("Top prioritized chunks", 10, 60, 28, 2)
    concurrency = st.slider("Concurrency (rate-limit risk)", 1, 12, 6, 1)
    min_rels = st.slider("Min relations target", 10, 80, 35, 1)

# ---------- Neo4j ----------
sync_neo4j = st.sidebar.checkbox("🔗 Sync to Neo4j Database", value=False)

neo4j_url = neo4j_user = neo4j_pass = None
if sync_neo4j:
    with st.sidebar.expander("Neo4j Credentials", expanded=True):
        st.caption("If left empty, the pipeline will use your environment/.env defaults if you set them.")
        neo4j_url = st.text_input("Neo4j URL", value=os.getenv("NEO4J_URL", "bolt://localhost:7687"))
        neo4j_user = st.text_input("Neo4j Username", value=os.getenv("NEO4J_USERNAME", "neo4j"))
        neo4j_pass = st.text_input("Neo4j Password", value=os.getenv("NEO4J_PASSWORD", ""), type="password")

st.sidebar.divider()

# ---------------- File / Text Input ----------------
source = None
is_path = False
temp_pdf_path = None

if "pdf" in input_method.lower():
    uploaded_file = st.sidebar.file_uploader(
        label="Upload Research Paper (PDF or TXT)",
        type=["pdf", "txt"]
    )

    if uploaded_file is not None:
        if uploaded_file.type == "application/pdf" or uploaded_file.name.lower().endswith(".pdf"):
            # ✅ unique temp file (no collisions between users/sessions)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(uploaded_file.read())
                temp_pdf_path = tmp.name
            source = temp_pdf_path
            is_path = True
        else:
            # TXT - robust decode
            raw = uploaded_file.read()
            try:
                source = raw.decode("utf-8")
            except UnicodeDecodeError:
                source = raw.decode("latin-1", errors="ignore")
            is_path = False
else:
    source = st.sidebar.text_area("Paste your research abstract or full text:", height=300)
    is_path = False

# ---------------- Helpers ----------------
def build_pipeline_config() -> PipelineConfig:
    cfg = PipelineConfig(
        max_total_chunks=max_chunks,
        prioritize_top_k=top_k,
        max_concurrent_chunks=concurrency,
        min_relationships_target=min_rels,
    )
    if sync_neo4j and neo4j_url and neo4j_user is not None:
        cfg = PipelineConfig(
            max_total_chunks=max_chunks,
            prioritize_top_k=top_k,
            max_concurrent_chunks=concurrency,
            min_relationships_target=min_rels,
            neo4j_url=neo4j_url,
            neo4j_user=neo4j_user,
            neo4j_password=neo4j_pass or "",
        )
    return cfg

def read_html(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()

# ---------------- Generate KG ----------------
if source:
    if st.sidebar.button("🚀 Generate Knowledge Graph"):
        cfg = build_pipeline_config()

        with st.spinner("🧠 Analyzing full paper content... This may take a moment to ensure complete coverage."):
            try:
                # Unique output file per run (avoid stale graph)
                # The pipeline saves as knowledge_graph.html; we rename after generation.
                run_id = str(int(time.time()))
                expected_html = "knowledge_graph.html"
                renamed_html = f"knowledge_graph_{run_id}.html"

                net, graph_docs, sync_status = generate_knowledge_graph(
                    source,
                    is_path=is_path,
                    sync_neo4j=sync_neo4j,
                    cfg=cfg
                )

                if graph_docs and len(graph_docs) > 0:
                    rel_count = len(graph_docs[0].relationships or [])
                    node_count = len(graph_docs[0].nodes or [])
                    st.success(f"✨ Knowledge graph generated with {node_count} nodes and {rel_count} relationships!")

                    if sync_neo4j:
                        if sync_status:
                            st.info("✅ Successfully synced to Neo4j.")
                        else:
                            st.warning("⚠️ Neo4j sync failed. Check your credentials/.env and Neo4j status.")

                    # Show graph
                    if os.path.exists(expected_html):
                        # Rename to avoid overwrite conflicts / stale downloads
                        try:
                            os.replace(expected_html, renamed_html)
                        except Exception:
                            renamed_html = expected_html

                        html_content = read_html(renamed_html)
                        col1, col2 = st.columns([4, 1])

                        with col1:
                            components.html(html_content, height=800, scrolling=True)

                        with col2:
                            st.subheader("📊 Graph Info")
                            st.info("Drag to move, scroll to zoom, hover nodes/edges for details.")
                            st.metric("Nodes", node_count)
                            st.metric("Relationships", rel_count)

                            with open(renamed_html, "rb") as file:
                                st.download_button(
                                    label="📥 Download HTML Graph",
                                    data=file,
                                    file_name="Data2Dash_knowledge_graph.html",
                                    mime="text/html"
                                )
                    else:
                        st.error("Graph HTML was not generated. Check logs.")

                else:
                    st.error("Failed to generate graph documents (empty result). Try increasing max chunks or lowering concurrency.")

            except Exception as e:
                st.error(f"❌ Error: {e}")
                import traceback
                with st.expander("Show traceback"):
                    st.code(traceback.format_exc())

            finally:
                # cleanup temp file
                if temp_pdf_path and os.path.exists(temp_pdf_path):
                    try:
                        os.remove(temp_pdf_path)
                    except Exception:
                        pass
else:
    st.info("👈 Please upload a file or enter text in the sidebar to get started.")

# ---------------- Footer ----------------
st.markdown("---")
st.markdown("**Data2Dash** — Research & Knowledge Extraction")
