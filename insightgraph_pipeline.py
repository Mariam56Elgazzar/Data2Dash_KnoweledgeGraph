# ===============================
# InsightGraph – Production Pipeline
# ===============================

from __future__ import annotations

import os
import re
import json
import time
import random
import logging
import asyncio
from dataclasses import dataclass
from typing import List, Tuple, Optional, Iterable, Dict

from dotenv import load_dotenv

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage
from langchain_groq import ChatGroq

from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_experimental.graph_transformers.llm import GraphDocument

from langchain_community.document_loaders import PyPDFLoader
from langchain_community.graphs import Neo4jGraph
from langchain_community.graphs.graph_document import Node, Relationship

# Your preprocessing module (use your production-grade one if you replaced it)
from preprocessing import preprocess_text, split_by_sections, sliding_window_chunks, page_based_chunks


# ----------------------------
# Logging
# ----------------------------
LOGGER = logging.getLogger("insightgraph")
if not LOGGER.handlers:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")


# ----------------------------
# Config
# ----------------------------
@dataclass(frozen=True)
class PipelineConfig:
    # LLM
    model_name: str = "llama-3.1-8b-instant"
    temperature: float = 0.0
    max_chunk_chars_for_llm: int = 6000  # hard cap per request (Groq payload safety)

    # Chunking bounds (IMPORTANT for cost/rate-limits)
    max_total_chunks: int = 40           # total chunks sent to transformer
    max_direct_passes: int = 8           # passes for direct fallback extractor
    prioritize_top_k: int = 28           # after scoring, keep only top-k chunks

    # Concurrency / retries
    max_concurrent_chunks: int = 6       # tune up/down depending on Groq limits
    max_retries: int = 3
    retry_base_delay: float = 1.0        # seconds

    # Fallback thresholds
    min_relationships_target: int = 35   # if below, do extra fallback

    # Neo4j (optional)
    neo4j_url: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: str = "password"


# ----------------------------
# Schema
# ----------------------------
ALLOWED_NODES = [
    "Concept", "Method", "Metric", "Dataset", "Result", "Challenge",
    "Author", "Organization", "Hyperparameter", "Technique",
    "Contribution", "Theory", "Process", "Task", "Baseline",
    "Advantage", "Observation", "Limitation", "FutureWork",
    "Model", "Component", "Benchmark", "Architecture", "Algorithm",
    "Publication", "System"
]

ALLOWED_RELATIONSHIPS = [
    "RELATED_TO", "USES", "CONTAINS", "COMPARED_TO", "ILLUSTRATES",
    "TRAINED_ON", "COMPARED_TO_ON", "USED_FOR", "IMPLEMENTS", "EVALUATES",
    "ACHIEVES", "ADDRESSES", "RESULTS_IN", "PART_OF", "CONTRIBUTES_TO",
    "IMPROVES", "SUPPORTS", "DEPENDS_ON", "DESCRIBED_IN", "PROPOSES",
    "OBSERVED_IN", "EXTENDS", "LIMITS", "INTRODUCES", "CITES"
]

RESEARCH_PAPER_INSTRUCTIONS = (
    "Extract ALL entities and relations. Cover every important detail: models, methods, datasets, metrics, "
    "baselines, components, architectures, results, contributions, limitations, comparisons, authors, "
    "algorithms, techniques, tasks, benchmarks, key findings, hyperparameters. "
    "Output a JSON array. Each object: head, head_type, relation, tail, tail_type. "
    "Example: [{\"head\":\"BERT\",\"head_type\":\"Model\",\"relation\":\"TRAINED_ON\",\"tail\":\"Wikipedia\",\"tail_type\":\"Dataset\"}]. "
    "Extract 30-70 relations when possible. Output ONLY the JSON array."
)


# ----------------------------
# Env / LLM
# ----------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

if not GROQ_API_KEY:
    LOGGER.warning("GROQ_API_KEY is missing. Set it in .env or environment variables.")

def build_llm(cfg: PipelineConfig) -> ChatGroq:
    # ✅ This is critical: pass api_key explicitly to avoid silent auth failures
    return ChatGroq(
        api_key=GROQ_API_KEY,
        temperature=cfg.temperature,
        model_name=cfg.model_name,
    )

def build_transformer(llm: ChatGroq) -> LLMGraphTransformer:
    return LLMGraphTransformer(
        llm=llm,
        allowed_nodes=ALLOWED_NODES,
        allowed_relationships=ALLOWED_RELATIONSHIPS,
        strict_mode=False,
        node_properties=False,
        relationship_properties=False,
        ignore_tool_usage=True,  # important for Groq/Llama stability
        additional_instructions=RESEARCH_PAPER_INSTRUCTIONS,
    )


# ----------------------------
# PDF Loader
# ----------------------------
def load_pdf_text(pdf_path: str, with_page_markers: bool = True) -> str:
    loader = PyPDFLoader(pdf_path)
    docs = loader.load()
    if not docs:
        return ""

    if not with_page_markers:
        return "\n\n".join((d.page_content or "").strip() for d in docs if (d.page_content or "").strip())

    parts = []
    for i, d in enumerate(docs):
        page_num = (getattr(d, "metadata", {}) or {}).get("page", i + 1)
        content = (d.page_content or "").strip()
        if content:
            parts.append(f"\n\n--- Page {page_num} ---\n\n{content}")
    return "\n\n".join(parts).strip()


# ----------------------------
# Chunk Prioritization (bounded)
# ----------------------------
def prioritize_chunks(chunks: List[str]) -> List[str]:
    priority_keywords = [
        "method", "architecture", "model", "transformer", "attention",
        "experiment", "results", "evaluation", "dataset", "benchmark",
        "training", "loss", "approach", "encoder", "decoder",
        "et al", "proposed", "compared", "achieves", "outperforms",
        "baseline", "metric", "accuracy", "f1", "bleu",
        "ablation", "limitation", "hyperparameter", "contribution"
    ]
    scored = [(sum(k in c.lower() for k in priority_keywords), c) for c in chunks]
    scored.sort(reverse=True, key=lambda x: x[0])
    return [c for _, c in scored]


def make_bounded_chunks(text: str, cfg: PipelineConfig) -> List[str]:
    # Keep your “full coverage” idea, but bound it.
    page_chunks = page_based_chunks(text, min_page_chars=120)
    section_chunks = split_by_sections(text, max_chunk_size=2600, overlap=900)
    sw_chunks = sliding_window_chunks(text, window_size=2400, step=700, max_chunks=40)

    # Dedup while preserving order
    all_chunks = []
    seen = set()
    for c in (page_chunks + section_chunks + sw_chunks):
        cc = (c or "").strip()
        if len(cc) < 200:
            continue
        if cc in seen:
            continue
        seen.add(cc)
        all_chunks.append(cc)

    # Prioritize and keep top-k
    all_chunks = prioritize_chunks(all_chunks)
    all_chunks = all_chunks[: cfg.prioritize_top_k]

    # Hard cap (for cost)
    return all_chunks[: cfg.max_total_chunks]


# ----------------------------
# Direct LLM Fallback
# ----------------------------
DIRECT_PROMPT = (
    "Extract ALL entities and relations. Cover models, datasets, metrics, baselines, methods, components, "
    "results, contributions, limitations, comparisons, authors, techniques.\n"
    "Each object: {\"head\":\"entity1\",\"head_type\":\"Model\",\"relation\":\"USES\",\"tail\":\"entity2\",\"tail_type\":\"Component\"}\n"
    "Node types: Model, Method, Dataset, Metric, Component, Concept, Author, Result, Baseline, Technique, "
    "Architecture, Task, Algorithm, Benchmark, Observation, Limitation, Contribution.\n"
    "Relation types: USES, CONTAINS, RELATED_TO, PART_OF, COMPARED_TO, TRAINED_ON, EVALUATES, IMPROVES, "
    "IMPLEMENTS, ACHIEVES, ADDRESSES, RESULTS_IN, PROPOSES, EXTENDS, DEPENDS_ON, SUPPORTS, ILLUSTRATES, "
    "CONTRIBUTES_TO, INTRODUCES, OBSERVED_IN, LIMITS, CITES.\n"
    "Extract 20-40 relations. Output ONLY valid JSON array (no markdown)."
)

def _extract_json_array(raw: str) -> List[dict]:
    raw = (raw or "").strip()

    # strip fenced blocks
    m = re.search(r"```(?:json)?\s*([\s\S]*?)\s*```", raw)
    if m:
        raw = m.group(1).strip()

    # find the first '[' and cut to matching ']'
    start = raw.find("[")
    if start < 0:
        return []

    depth = 0
    end = None
    for i, ch in enumerate(raw[start:], start):
        if ch == "[":
            depth += 1
        elif ch == "]":
            depth -= 1
            if depth == 0:
                end = i + 1
                break

    if end is None:
        return []

    candidate = raw[start:end]
    try:
        out = json.loads(candidate)
        return out if isinstance(out, list) else []
    except Exception:
        return []


def _direct_extract_to_graph(llm: ChatGroq, text: str, cfg: PipelineConfig) -> Tuple[List[Node], List[Relationship]]:
    try:
        msg = llm.invoke([HumanMessage(content=DIRECT_PROMPT + "\n\nText:\n" + text[: cfg.max_chunk_chars_for_llm])])
        raw = msg.content if hasattr(msg, "content") else str(msg)
        parsed = _extract_json_array(raw)

        nodes_set = set()
        rels: List[Relationship] = []

        for rel in parsed:
            if not isinstance(rel, dict):
                continue
            h = (rel.get("head") or "").strip()
            t = (rel.get("tail") or "").strip()
            r = (rel.get("relation") or "").strip()
            if not (h and t and r):
                continue

            ht = (rel.get("head_type") or "Concept").strip()
            tt = (rel.get("tail_type") or "Concept").strip()
            r = r.upper().replace(" ", "_")

            nodes_set.add((h, ht))
            nodes_set.add((t, tt))
            rels.append(Relationship(
                source=Node(id=h, type=ht),
                target=Node(id=t, type=tt),
                type=r,
            ))

        nodes = [Node(id=n, type=t) for (n, t) in nodes_set]
        return nodes, rels

    except Exception as e:
        LOGGER.warning("Direct extraction failed: %s", e)
        return [], []


# ----------------------------
# Async Extraction
# ----------------------------
async def _process_one_chunk(
    i: int,
    chunk: str,
    transformer: LLMGraphTransformer,
    cfg: PipelineConfig,
    semaphore: asyncio.Semaphore,
):
    async with semaphore:
        docs = [Document(page_content=chunk[: cfg.max_chunk_chars_for_llm])]
        for attempt in range(cfg.max_retries):
            try:
                gd = await transformer.aconvert_to_graph_documents(docs)
                return i, gd
            except Exception as e:
                if attempt == cfg.max_retries - 1:
                    LOGGER.warning("Chunk %d failed after %d retries: %s", i + 1, cfg.max_retries, e)
                    return i, []
                # exponential backoff + jitter
                delay = cfg.retry_base_delay * (2 ** attempt) + random.random() * 0.2
                await asyncio.sleep(delay)
    return i, []


async def extract_graph_data_from_chunks(
    chunks: List[str],
    transformer: LLMGraphTransformer,
    llm: ChatGroq,
    cfg: PipelineConfig,
) -> List[GraphDocument]:
    sem = asyncio.Semaphore(cfg.max_concurrent_chunks)
    tasks = [_process_one_chunk(i, ch, transformer, cfg, sem) for i, ch in enumerate(chunks)]
    results = await asyncio.gather(*tasks, return_exceptions=True)

    nodes: List[Node] = []
    rels: List[Relationship] = []

    for r in results:
        if isinstance(r, Exception):
            LOGGER.warning("Task exception: %s", r)
            continue
        _, gdocs = r
        for g in gdocs or []:
            nodes.extend(getattr(g, "nodes", []) or [])
            rels.extend(getattr(g, "relationships", []) or [])

    combined = "\n\n".join(chunks)

    # Fallback transformer pass if too few relations
    if len(rels) < cfg.min_relationships_target and len(combined) > 800:
        LOGGER.info("Few relations (%d). Running combined-pass transformer fallback...", len(rels))
        for offset in range(0, min(len(combined), 2 * cfg.max_chunk_chars_for_llm), cfg.max_chunk_chars_for_llm):
            passage = combined[offset: offset + cfg.max_chunk_chars_for_llm]
            try:
                gdocs = await transformer.aconvert_to_graph_documents([Document(page_content=passage)])
                for g in gdocs or []:
                    nodes.extend(g.nodes)
                    rels.extend(g.relationships)
            except Exception as e:
                LOGGER.warning("Transformer fallback failed: %s", e)

    # Direct extraction supplement (bounded)
    if len(combined) > 500:
        LOGGER.info("Running bounded direct LLM extraction supplement...")
        step = 2200
        size = 5400
        passes = 0
        for offset in range(0, len(combined), step):
            if passes >= cfg.max_direct_passes:
                break
            passage = combined[offset: offset + size]
            if len(passage.strip()) < 250:
                continue
            dn, dr = _direct_extract_to_graph(llm, passage, cfg)
            nodes.extend(dn)
            rels.extend(dr)
            passes += 1

    return [GraphDocument(
        nodes=nodes,
        relationships=rels,
        source=Document(page_content="Merged chunks")
    )]


# ----------------------------
# Post-processing: canonicalization + dedup + endpoint integrity
# ----------------------------
def _canon(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())

def postprocess_graph(graph_doc: GraphDocument) -> GraphDocument:
    # Canonicalize node IDs
    canonical_nodes: Dict[Tuple[str, str], Node] = {}
    for n in list(graph_doc.nodes or []):
        nid = _canon(getattr(n, "id", ""))
        ntype = _canon(getattr(n, "type", "") or "Concept")
        if not nid:
            continue
        key = (nid.lower(), ntype)
        if key not in canonical_nodes:
            canonical_nodes[key] = Node(id=nid, type=ntype)

    id_index = {k[0]: v for k, v in canonical_nodes.items()}

    # Dedup relationships
    seen_edges = set()
    unique_edges: List[Relationship] = []

    for r in list(graph_doc.relationships or []):
        sid = _canon(getattr(r.source, "id", ""))
        tid = _canon(getattr(r.target, "id", ""))
        rtype = _canon(getattr(r, "type", "") or "RELATED_TO").upper().replace(" ", "_")
        if not sid or not tid:
            continue
        sk, tk = sid.lower(), tid.lower()
        ek = (sk, tk, rtype)
        if ek in seen_edges:
            continue
        seen_edges.add(ek)

        # ensure endpoints exist
        if sk not in id_index:
            id_index[sk] = Node(id=sid, type=getattr(r.source, "type", None) or "Concept")
        if tk not in id_index:
            id_index[tk] = Node(id=tid, type=getattr(r.target, "type", None) or "Concept")

        unique_edges.append(Relationship(source=id_index[sk], target=id_index[tk], type=rtype))

    graph_doc.nodes = list({(n.id, n.type): n for n in id_index.values()}.values())
    graph_doc.relationships = unique_edges
    return graph_doc


# ----------------------------
# Visualization (keep your PyVis code if you want)
# ----------------------------
from pyvis.network import Network

def visualize_graph(graph_documents: List[GraphDocument], output_file: str = "knowledge_graph.html"):
    if not graph_documents:
        return None

    net = Network(height="1000px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222")
    graph = graph_documents[0]

    for node in graph.nodes:
        net.add_node(node.id, label=node.id, title=node.type, color="#b9d9ea")

    for rel in graph.relationships:
        net.add_edge(rel.source.id, rel.target.id, label=rel.type, color="#97c2fc")

    net.set_options("""
    {
      "physics": {
        "forceAtlas2Based": {
          "gravitationalConstant": -50,
          "centralGravity": 0.01,
          "springLength": 110,
          "springConstant": 0.08
        },
        "maxVelocity": 50,
        "solver": "forceAtlas2Based",
        "timestep": 0.35,
        "stabilization": { "enabled": true, "iterations": 900 }
      },
      "interaction": {
        "navigationButtons": true,
        "keyboard": true,
        "hover": true,
        "zoomView": true
      }
    }
    """)
    net.save_graph(output_file)
    return net


# ----------------------------
# Neo4j Sync (real connection)
# ----------------------------
def sync_to_neo4j(graph_documents: List[GraphDocument], cfg: PipelineConfig) -> bool:
    try:
        graph = Neo4jGraph(url=cfg.neo4j_url, username=cfg.neo4j_user, password=cfg.neo4j_password)
        graph.add_graph_documents(graph_documents)
        return True
    except Exception as e:
        LOGGER.warning("Neo4j Error: %s", e)
        return False


# ----------------------------
# Safe async runner (works in Streamlit/Jupyter too)
# ----------------------------
def run_async(coro):
    """
    Safely run async code in environments that may already have an event loop.
    - In normal Python: uses asyncio.run
    - In running loop (Streamlit/Jupyter): creates a new task and waits
    """
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)

    # Running loop exists
    return loop.run_until_complete(coro)  # may fail in some notebooks


def generate_knowledge_graph(source: str, is_path: bool = True, sync_neo4j: bool = False, cfg: Optional[PipelineConfig] = None):
    cfg = cfg or PipelineConfig()

    llm = build_llm(cfg)
    transformer = build_transformer(llm)

    # 1) Load
    text = load_pdf_text(source, with_page_markers=True) if is_path else (source or "")
    if not text.strip():
        LOGGER.warning("No text extracted.")
        return None, [GraphDocument(nodes=[], relationships=[], source=Document(page_content=""))], None

    # 2) Preprocess
    text = preprocess_text(text)

    # 3) Chunk (bounded)
    chunks = make_bounded_chunks(text, cfg)
    if not chunks:
        LOGGER.warning("No chunks produced.")
        return None, [GraphDocument(nodes=[], relationships=[], source=Document(page_content=""))], None

    LOGGER.info("Processing %d chunks (bounded)", len(chunks))

    # 4) Extract
    # If you're in Streamlit and this errors, replace run_async() with an "await" call in your Streamlit async flow.
    try:
        graph_docs = asyncio.run(extract_graph_data_from_chunks(chunks, transformer, llm, cfg))
    except RuntimeError:
        # event loop already running
        graph_docs = run_async(extract_graph_data_from_chunks(chunks, transformer, llm, cfg))

    # 5) Postprocess
    graph_docs[0] = postprocess_graph(graph_docs[0])

    LOGGER.info("Nodes: %d", len(graph_docs[0].nodes))
    LOGGER.info("Relations: %d", len(graph_docs[0].relationships))

    # 6) Visualize
    net = visualize_graph(graph_docs)

    # 7) Neo4j optional
    sync_success = sync_to_neo4j(graph_docs, cfg) if sync_neo4j else None

    return net, graph_docs, sync_success


def generate_knowledge_graph_from_pdf(pdf_path: str):
    return generate_knowledge_graph(pdf_path, is_path=True)
