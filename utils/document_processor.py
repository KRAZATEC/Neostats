"""
NeoStats AI Chatbot — Document Processing Utilities
Handles file parsing, text chunking, and RAG pipeline orchestration.
"""

import io
import logging
import os
import re
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ─── Text Extraction ───────────────────────────────────────────────────────────

def extract_text_from_pdf(file_bytes: bytes) -> str:
    """Extract raw text from a PDF file."""
    try:
        import pdfplumber
        with pdfplumber.open(io.BytesIO(file_bytes)) as pdf:
            pages = []
            for page in pdf.pages:
                text = page.extract_text()
                if text:
                    pages.append(text.strip())
            return "\n\n".join(pages)
    except ImportError:
        # Fallback to pypdf
        try:
            import pypdf
            reader = pypdf.PdfReader(io.BytesIO(file_bytes))
            return "\n\n".join(
                p.extract_text() for p in reader.pages if p.extract_text()
            )
        except ImportError:
            raise ImportError("Install pdfplumber or pypdf: pip install pdfplumber")
    except Exception as e:
        logger.error(f"PDF extraction error: {e}")
        raise


def extract_text_from_docx(file_bytes: bytes) -> str:
    """Extract raw text from a DOCX file."""
    try:
        import docx
        doc = docx.Document(io.BytesIO(file_bytes))
        return "\n".join(p.text for p in doc.paragraphs if p.text.strip())
    except ImportError:
        raise ImportError("Install python-docx: pip install python-docx")
    except Exception as e:
        logger.error(f"DOCX extraction error: {e}")
        raise


def extract_text_from_csv(file_bytes: bytes) -> str:
    """Convert CSV rows into readable text blocks."""
    try:
        import csv
        text = io.StringIO(file_bytes.decode("utf-8", errors="replace"))
        reader = csv.DictReader(text)
        rows = []
        for i, row in enumerate(reader):
            row_text = " | ".join(f"{k}: {v}" for k, v in row.items() if v)
            rows.append(f"Row {i+1}: {row_text}")
        return "\n".join(rows)
    except Exception as e:
        logger.error(f"CSV extraction error: {e}")
        raise


def extract_text_from_excel(file_bytes: bytes) -> str:
    """Extract text from Excel (.xlsx, .xls) files using pandas."""
    try:
        import pandas as pd
        df = pd.read_excel(io.BytesIO(file_bytes))
        
        # Convert each row to a readable text block
        rows = []
        for i, row in df.iterrows():
            row_dict = row.to_dict()
            row_text = " | ".join(f"{k}: {v}" for k, v in row_dict.items() if pd.notna(v))
            rows.append(f"Row {i+1}: {row_text}")
            
        return "\n".join(rows)
    except ImportError:
        raise ImportError("Install pandas and openpyxl: pip install pandas openpyxl")
    except Exception as e:
        logger.error(f"Excel extraction error: {e}")
        raise


def extract_text(file_bytes: bytes, filename: str) -> str:
    """
    Route extraction to the appropriate parser based on file extension.

    Args:
        file_bytes: Raw file bytes
        filename: Original filename (used to detect extension)

    Returns:
        Extracted text string
    """
    ext = Path(filename).suffix.lower()
    extractors = {
        ".pdf": extract_text_from_pdf,
        ".docx": extract_text_from_docx,
        ".doc": extract_text_from_docx,
        ".csv": extract_text_from_csv,
        ".xlsx": extract_text_from_excel,
        ".xls": extract_text_from_excel,
        ".txt": lambda b: b.decode("utf-8", errors="replace"),
        ".md": lambda b: b.decode("utf-8", errors="replace"),
    }
    if ext not in extractors:
        raise ValueError(f"Unsupported file type: {ext}")
    try:
        return extractors[ext](file_bytes)
    except Exception as e:
        logger.error(f"Text extraction failed for {filename}: {e}")
        raise


# ─── Text Chunking ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize whitespace and remove junk characters."""
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)  # Remove non-ASCII
    return text.strip()


def chunk_text(
    text: str,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
    source: str = "unknown",
) -> list[dict]:
    """
    Split text into overlapping chunks for vector indexing.

    Args:
        text: Full document text
        chunk_size: Characters per chunk
        chunk_overlap: Overlap between consecutive chunks
        source: Source document name

    Returns:
        List of chunk dicts: {text, source, chunk_id, char_start, char_end}
    """
    text = clean_text(text)
    if not text:
        return []

    chunks = []
    start = 0
    chunk_id = 0

    while start < len(text):
        end = start + chunk_size

        # Try to break at sentence boundary
        if end < len(text):
            # Look for sentence-ending punctuation
            boundary = max(
                text.rfind(". ", start, end),
                text.rfind("! ", start, end),
                text.rfind("? ", start, end),
                text.rfind("\n", start, end),
            )
            if boundary > start + chunk_size // 2:
                end = boundary + 1

        chunk_text_val = text[start:end].strip()
        if len(chunk_text_val) > 50:  # Skip very small fragments
            chunks.append({
                "text": chunk_text_val,
                "source": source,
                "chunk_id": chunk_id,
                "char_start": start,
                "char_end": end,
            })
            chunk_id += 1

        start = end - chunk_overlap if end < len(text) else len(text)

    logger.info(f"Chunked '{source}' into {len(chunks)} chunks")
    return chunks


# ─── RAG Orchestration ─────────────────────────────────────────────────────────

def build_rag_context(retrieved_chunks: list[dict], max_length: int = 3000) -> str:
    """
    Format retrieved chunks into a context string for the LLM prompt.

    Args:
        retrieved_chunks: List of chunks from vector search
        max_length: Max total character length of context

    Returns:
        Formatted context string
    """
    if not retrieved_chunks:
        return ""

    parts = ["📚 **Relevant Knowledge Base Context:**\n"]
    total_len = 0

    for i, chunk in enumerate(retrieved_chunks, 1):
        score = chunk.get("score", 0)
        source = chunk.get("source", "unknown")
        text = chunk.get("text", "")

        entry = f"\n[Source {i}: {source} | Relevance: {score:.2f}]\n{text}\n"
        if total_len + len(entry) > max_length:
            break
        parts.append(entry)
        total_len += len(entry)

    parts.append("\n---\n")
    return "".join(parts)


def process_uploaded_file(
    file_bytes: bytes,
    filename: str,
    vector_store,
    chunk_size: int = 512,
    chunk_overlap: int = 64,
) -> dict:
    """
    Full pipeline: extract text → chunk → embed → index.

    Args:
        file_bytes: Raw uploaded file bytes
        filename: Original file name
        vector_store: FAISSVectorStore instance
        chunk_size: Chunk character length
        chunk_overlap: Chunk overlap length

    Returns:
        Dict with status info: {success, chunks_added, char_count, error}
    """
    try:
        # 1. Extract text
        raw_text = extract_text(file_bytes, filename)
        if not raw_text.strip():
            return {"success": False, "error": "No text could be extracted from file."}

        # 2. Chunk text
        chunks = chunk_text(raw_text, chunk_size, chunk_overlap, source=filename)
        if not chunks:
            return {"success": False, "error": "Document produced no usable chunks."}

        # 3. Embed and index
        added = vector_store.add_documents(chunks)

        return {
            "success": True,
            "chunks_added": added,
            "char_count": len(raw_text),
            "error": None,
        }
    except Exception as e:
        logger.error(f"File processing pipeline failed for {filename}: {e}")
        return {"success": False, "chunks_added": 0, "char_count": 0, "error": str(e)}
