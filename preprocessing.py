# ===============================
# preprocessing.py (production-grade)
# ===============================
"""
Production-grade preprocessing for scientific / academic documents
to improve downstream LLM extraction and Knowledge Graph quality.

Key goals:
- Preserve semantic structure (sections/newlines) while removing noise.
- Stop before non-informative reference material.
- Normalize common abbreviations to reduce node fragmentation.
- Provide robust chunking (section-aware + sliding window + page markers).
- Be configurable, testable, and safe on messy PDF-to-text output.

Dependencies: standard library only.
"""

from __future__ import annotations

import os
import re
import unicodedata
import logging
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional, Tuple


# ----------------------------
# Logging
# ----------------------------
LOGGER = logging.getLogger(__name__)


# ----------------------------
# Config
# ----------------------------
DEFAULT_ENTITY_MAP: Dict[str, str] = {
    "llms": "Large Language Models",
    "llm": "Large Language Model",
    "kg": "Knowledge Graph",
    "kgs": "Knowledge Graphs",
    "nlp": "Natural Language Processing",
    "gnn": "Graph Neural Network",
    "transformer": "Transformer Architecture",
    "cnn": "Convolutional Neural Network",
    "rnn": "Recurrent Neural Network",
}

DEFAULT_STOP_HEADINGS = (
    "references",
    "bibliography",
    "acknowledgment",
    "acknowledgements",
    "appendix",
    "supplementary material",
    "author contributions",
    "ethics statement",
    "data availability",
    "conflict of interest",
    "funding",
)

# A broad, paper-friendly section header set.
DEFAULT_SECTION_HEADINGS = (
    "Abstract",
    "Introduction",
    "Background",
    "Related Work",
    "Preliminaries",
    "Methodology",
    "Methods",
    "Approach",
    "Model Architecture",
    "Model",
    "Architecture",
    "Implementation",
    "Training",
    "Datasets",
    "Benchmarks",
    "Experimental Setup",
    "Setup",
    "Experiments",
    "Results",
    "Evaluation",
    "Analysis",
    "Ablation",
    "Discussion",
    "Limitations",
    "Future Work",
    "Conclusion",
    "Contributions",
    "Key Findings",
    "Findings",
    "Proposed Method",
)


@dataclass(frozen=True)
class PreprocessConfig:
    # Text normalization
    unicode_normalize: bool = True                 # NFKC normalization
    strip_null_bytes: bool = True                  # remove \x00
    normalize_newlines: bool = True                # \r\n -> \n
    keep_double_newlines: bool = True              # collapse 3+ newlines to 2

    # Noise removal / cleanup
    remove_numeric_citations: bool = True          # [1], [1,2], [1–3]
    remove_year_only_parens: bool = True           # (2020)
    remove_bullets: bool = True
    remove_decorative_lines: bool = True           # -----, =====
    fix_hyphenated_linebreaks: bool = True         # "atten-\n tion" -> "attention"
    remove_inline_latex: bool = False              # optional: remove $...$ (can hurt math-heavy papers)
    remove_display_latex: bool = False             # optional: remove \[...\] / $$...$$

    # Stop sections (truncate)
    stop_headings: Tuple[str, ...] = DEFAULT_STOP_HEADINGS

    # Entity normalization
    entity_map: Dict[str, str] = field(default_factory=lambda: dict(DEFAULT_ENTITY_MAP))

    # Chunking defaults
    max_chunk_size: int = 3500
    overlap: int = 450

    sliding_window_size: int = 2800
    sliding_step: int = 900
    sliding_max_chunks: int = 30

    # Page chunking
    min_page_chars: int = 150

    # Section header detection
    section_headings: Tuple[str, ...] = DEFAULT_SECTION_HEADINGS


# ----------------------------
# Regex compilation (performance + consistency)
# ----------------------------
def _compile_patterns(cfg: PreprocessConfig) -> Dict[str, re.Pattern]:
    # Numeric citations: [1], [1, 2], [1-3], [1–3], [1,2–5]
    numeric_citations = re.compile(
        r"\[(?:\s*\d+\s*(?:[-–]\s*\d+\s*)?)(?:\s*,\s*\d+\s*(?:[-–]\s*\d+\s*)?)*\]"
    )

    year_only_parens = re.compile(r"\(\s*\d{4}\s*\)")
    bullets = re.compile(r"[•▪■►◆▶●◦]")
    decorative_lines = re.compile(r"(?:-{3,}|={3,}|_{3,}|\*{3,})")
    spaces_tabs = re.compile(r"[ \t]+")
    many_newlines = re.compile(r"\n{3,}")

    # Hyphenation at line breaks: "atten-\n tion" -> "attention"
    # We only join if the hyphen is at end-of-line and next starts with a letter.
    hyphen_linebreak = re.compile(r"(\w)-\n(\w)")

    # Optional LaTeX removal
    inline_latex = re.compile(r"\$(?!\s)(.*?)(?<!\s)\$", flags=re.DOTALL)
    display_latex_1 = re.compile(r"\\\[(.*?)\\\]", flags=re.DOTALL)
    display_latex_2 = re.compile(r"\$\$(.*?)\$\$", flags=re.DOTALL)

    # Stop heading matcher (built from cfg.stop_headings)
    # matches "References" or "6. References" or "6 References" etc.
    stop_heading_union = "|".join(re.escape(h) for h in cfg.stop_headings)
    stop_heading = re.compile(rf"^(\d+\.?\s*)?({stop_heading_union})$", flags=re.IGNORECASE)

    # Section splitter: capture heading name so it appears in split results.
    section_union = "|".join(cfg.section_headings).replace(" ", r"\s+")
    section_regex = re.compile(
        rf"""
        (?m)^
        (?:\d+\.|[IVX]+\.|[IVX]+\b|\d+\b)?\s*
        ({section_union})
        \s*$
        """,
        flags=re.IGNORECASE | re.VERBOSE,
    )

    return {
        "numeric_citations": numeric_citations,
        "year_only_parens": year_only_parens,
        "bullets": bullets,
        "decorative_lines": decorative_lines,
        "spaces_tabs": spaces_tabs,
        "many_newlines": many_newlines,
        "hyphen_linebreak": hyphen_linebreak,
        "inline_latex": inline_latex,
        "display_latex_1": display_latex_1,
        "display_latex_2": display_latex_2,
        "stop_heading": stop_heading,
        "section_regex": section_regex,
    }


# ----------------------------
# Core Cleaning
# ----------------------------
def clean_text(text: str, cfg: Optional[PreprocessConfig] = None) -> str:
    """
    Basic text cleaning for scientific documents without destroying semantic structure.
    Keeps newlines (important for section detection and chunking).
    """
    cfg = cfg or PreprocessConfig()
    pat = _compile_patterns(cfg)

    if text is None:
        return ""

    if cfg.strip_null_bytes:
        text = text.replace("\x00", "")

    if cfg.normalize_newlines:
        text = text.replace("\r\n", "\n").replace("\r", "\n")

    if cfg.unicode_normalize:
        # NFKC is a practical choice for messy PDF text:
        # folds full-width chars, compatibility forms, etc.
        text = unicodedata.normalize("NFKC", text)

    # Fix hyphenation before whitespace normalization, but after newline normalization.
    if cfg.fix_hyphenated_linebreaks:
        text = pat["hyphen_linebreak"].sub(r"\1\2", text)

    # Remove citations / noise
    if cfg.remove_numeric_citations:
        text = pat["numeric_citations"].sub("", text)

    if cfg.remove_year_only_parens:
        text = pat["year_only_parens"].sub("", text)

    if cfg.remove_bullets:
        text = pat["bullets"].sub("", text)

    if cfg.remove_decorative_lines:
        text = pat["decorative_lines"].sub(" ", text)

    # Optional LaTeX removal (off by default)
    if cfg.remove_display_latex:
        text = pat["display_latex_1"].sub(" ", text)
        text = pat["display_latex_2"].sub(" ", text)
    if cfg.remove_inline_latex:
        text = pat["inline_latex"].sub(" ", text)

    # Normalize whitespace but keep newlines
    text = pat["spaces_tabs"].sub(" ", text)

    if cfg.keep_double_newlines:
        text = pat["many_newlines"].sub("\n\n", text)

    return text.strip()


# ----------------------------
# Remove Irrelevant Sections
# ----------------------------
def remove_irrelevant_sections(text: str, cfg: Optional[PreprocessConfig] = None) -> str:
    """
    Truncate non-informative tail sections like References/Appendix.
    Keeps Abstract through Conclusion; stops at the first stop heading.
    """
    cfg = cfg or PreprocessConfig()
    pat = _compile_patterns(cfg)

    if not text.strip():
        return ""

    lines = text.split("\n")
    filtered_lines: List[str] = []

    for line in lines:
        clean_line = line.strip().lower()

        # Heuristic: headers are short and usually alone on a line.
        if 0 < len(clean_line) < 60 and pat["stop_heading"].match(clean_line):
            break

        filtered_lines.append(line)

    return "\n".join(filtered_lines).strip()


# ----------------------------
# Entity Normalization
# ----------------------------
def normalize_entities(text: str, cfg: Optional[PreprocessConfig] = None) -> str:
    """
    Normalize common abbreviations to improve node consistency.
    Example: "LLMs" -> "Large Language Models"
    """
    cfg = cfg or PreprocessConfig()
    if not text.strip():
        return ""

    # Sort by length descending to avoid partial overlaps (e.g., llm before llms is bad)
    items = sorted(cfg.entity_map.items(), key=lambda kv: len(kv[0]), reverse=True)

    for short, full in items:
        # Word-boundary replacement, case-insensitive
        text = re.sub(rf"\b{re.escape(short)}\b", full, text, flags=re.IGNORECASE)

    return text


# ----------------------------
# Chunk Helpers
# ----------------------------
def _overlap_at_word_boundary(text: str, overlap: int) -> str:
    """Return last `overlap` characters aligned to a word boundary."""
    if overlap <= 0:
        return ""
    if len(text) <= overlap:
        return text.strip()
    tail = text[-overlap:].strip()
    first_space = tail.find(" ")
    return tail[first_space:].strip() if first_space > 0 else tail


def _split_at_word_boundary(text: str, max_len: int) -> List[str]:
    """Split into chunks of at most max_len, aligned to spaces when possible."""
    if max_len <= 0:
        return [text.strip()] if text.strip() else []
    chunks: List[str] = []
    t = text.strip()
    while len(t) > max_len:
        split_at = t.rfind(" ", 0, max_len + 1)
        if split_at <= 0:
            split_at = max_len
        chunks.append(t[:split_at].strip())
        t = t[split_at:].strip()
    if t:
        chunks.append(t)
    return chunks


# ----------------------------
# Section-aware Chunking
# ----------------------------
def split_by_sections(
    text: str,
    max_chunk_size: int = 3500,
    overlap: int = 450,
    cfg: Optional[PreprocessConfig] = None,
) -> List[str]:
    """
    Chunk text based on academic paper sections (numbered or unnumbered)
    with safe overlap for relationship continuity.
    """
    cfg = cfg or PreprocessConfig()
    pat = _compile_patterns(cfg)

    t = text.strip()
    if not t:
        return []

    parts = re.split(pat["section_regex"], t)
    final_chunks: List[str] = []
    current: List[str] = []

    # A lightweight heading detector for split results:
    heading_prefix = re.compile(
        r"^(abstract|introduction|background|related|prelim|method|approach|model|architect|implement|train|dataset|benchmark|experimental|setup|experiment|result|evaluat|analysis|ablation|discuss|limit|future|conclus|contribution|finding|proposed)",
        flags=re.IGNORECASE,
    )

    for part in parts:
        if not part:
            continue
        part = part.strip()
        if len(part) < 60 and heading_prefix.match(part):
            current.append(f"\n\n## {part}\n\n")
            continue

        current.append(part)
        full = " ".join(current).strip()

        if len(full) >= max_chunk_size:
            final_chunks.append(full)
            tail = _overlap_at_word_boundary(full, overlap)
            current = [tail] if tail else []

    if current:
        final_chunks.append(" ".join(current).strip())

    # Split any oversized chunks at word boundaries
    refined: List[str] = []
    for ch in final_chunks:
        if len(ch) > max_chunk_size + 400:
            refined.extend(_split_at_word_boundary(ch, max_chunk_size))
        else:
            refined.append(ch)

    # Filter tiny chunks
    return [c for c in refined if len(c.strip()) >= 200]


# ----------------------------
# Sliding Window Chunking
# ----------------------------
def sliding_window_chunks(
    text: str,
    window_size: int = 2800,
    step: int = 900,
    max_chunks: int = 30,
) -> List[str]:
    """
    Sliding window chunking for maximum recall (high overlap).
    Good when you want to not miss cross-boundary relations.
    """
    t = text.strip()
    if not t:
        return []
    if window_size <= 0 or step <= 0:
        return [t]

    chunks: List[str] = []
    start = 0
    while start < len(t) and len(chunks) < max_chunks:
        end = start + window_size
        chunk = t[start:end]
        if len(chunk.strip()) >= 200:
            chunks.append(chunk.strip())
        start += step
    return chunks


# ----------------------------
# Page-based Chunking
# ----------------------------
def page_based_chunks(text: str, min_page_chars: int = 150) -> List[str]:
    """
    Split text by page markers like '--- Page N ---'.
    Merges very short pages to avoid tiny chunks.
    """
    t = text.strip()
    if not t:
        return []

    pages = re.split(r"\n*--- Page \d+ ---\n*", t)
    pages = [p.strip() for p in pages if p.strip()]

    chunks: List[str] = []
    buffer = ""

    for p in pages:
        if buffer and (len(buffer) + len(p) < 4000):
            buffer = buffer + "\n\n" + p
        elif buffer:
            if len(buffer.strip()) >= min_page_chars:
                chunks.append(buffer.strip())
            buffer = p
        else:
            buffer = p

    if buffer and len(buffer.strip()) >= min_page_chars:
        chunks.append(buffer.strip())

    return chunks


# ----------------------------
# File Loading (txt/md only; intentionally minimal deps)
# ----------------------------
def load_text_file(path: str) -> str:
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        return f.read()


def load_document(path: str) -> str:
    """
    Load supported document types.

    Note: In production, PDF extraction is typically handled upstream
    (e.g., PyMuPDF/pdfplumber/Grobid), then passed here as text.
    """
    ext = os.path.splitext(path)[1].lower()
    if ext in (".txt", ".md"):
        return load_text_file(path)
    raise ValueError(f"Unsupported file type: {path}")


def load_documents_from_folder(folder_path: str) -> str:
    """
    Load and merge all supported documents in a folder.
    Skips unreadable files safely.
    """
    all_text: List[str] = []
    for name in os.listdir(folder_path):
        full_path = os.path.join(folder_path, name)
        if not os.path.isfile(full_path):
            continue
        try:
            all_text.append(load_document(full_path))
        except Exception as e:
            LOGGER.debug("Skipping %s (%s)", full_path, e)
            continue
    return "\n\n".join(all_text).strip()


# ----------------------------
# Full Preprocessing Pipeline
# ----------------------------
def preprocess_text(text: str, cfg: Optional[PreprocessConfig] = None) -> str:
    """
    Full preprocessing pipeline (safe for LLM graph extraction).
    """
    cfg = cfg or PreprocessConfig()
    t = clean_text(text, cfg=cfg)
    t = remove_irrelevant_sections(t, cfg=cfg)
    t = normalize_entities(t, cfg=cfg)
    return t


# ----------------------------
# Utility: choose chunking strategy
# ----------------------------
def make_chunks(
    text: str,
    strategy: str = "sections",
    cfg: Optional[PreprocessConfig] = None,
) -> List[str]:
    """
    Create chunks using one of:
      - "sections": section-aware (recommended default)
      - "sliding": sliding window (max recall)
      - "pages": page markers (if your upstream extractor adds them)
    """
    cfg = cfg or PreprocessConfig()
    t = text.strip()
    if not t:
        return []

    strategy = strategy.lower().strip()
    if strategy == "sections":
        return split_by_sections(t, cfg.max_chunk_size, cfg.overlap, cfg=cfg)
    if strategy == "sliding":
        return sliding_window_chunks(t, cfg.sliding_window_size, cfg.sliding_step, cfg.sliding_max_chunks)
    if strategy == "pages":
        return page_based_chunks(t, cfg.min_page_chars)

    raise ValueError(f"Unknown strategy: {strategy}")


# ----------------------------
# Example CLI usage (optional)
# ----------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    import argparse

    parser = argparse.ArgumentParser(description="Preprocess and chunk scientific text documents.")
    parser.add_argument("path", help="Path to .txt/.md file or a folder")
    parser.add_argument("--strategy", default="sections", choices=["sections", "sliding", "pages"])
    parser.add_argument("--print-chunks", action="store_true", help="Print chunks to stdout")
    args = parser.parse_args()

    if os.path.isdir(args.path):
        raw = load_documents_from_folder(args.path)
    else:
        raw = load_document(args.path)

    cfg = PreprocessConfig()
    processed = preprocess_text(raw, cfg=cfg)
    chunks = make_chunks(processed, strategy=args.strategy, cfg=cfg)

    print(f"Processed chars: {len(processed)}")
    print(f"Chunks: {len(chunks)}")
    if args.print_chunks:
        for i, c in enumerate(chunks, 1):
            print("\n" + "=" * 30)
            print(f"CHUNK {i} ({len(c)} chars)")
            print("=" * 30)
            print(c)
