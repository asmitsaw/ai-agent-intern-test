"""
agent/rag.py — Document indexing and retrieval.

Indexing  (run once via scripts/build_index.py):
  1. Parse YAML front matter from each .md file.
  2. Split into token-bounded chunks with heading tracking.
  3. Embed with OpenAI text-embedding-3-small.
  4. Store in ChromaDB with rich metadata.

Retrieval:
  1. Embed the query.
  2. Query ChromaDB for top-K candidates.
  3. Filter / penalise:
       • audience == "internal"  → excluded entirely
       • status == "superseded"  → excluded unless nothing else found
       • policy_authority == "official" + status == "active" → boosted rank
  4. Return chunks with metadata for citation.
"""
from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import frontmatter  # python-frontmatter
import tiktoken

import chromadb
from chromadb.utils import embedding_functions

from agent.config import (
    CHROMA_COLLECTION,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    EMBEDDING_MODEL,
    EMBEDDING_PROVIDER,
    KNOWLEDGE_BASE_PATH,
    LLM_API_KEY,
    LLM_BASE_URL,
    RETRIEVAL_TOP_K,
    VECTOR_STORE_PATH,
)

# ── Tokeniser (cl100k_base works for modern embedding models) ─────────────────
_enc = tiktoken.get_encoding("cl100k_base")


def _token_len(text: str) -> int:
    return len(_enc.encode(text))


# ── ChromaDB client & embedding function ─────────────────────────────────────

def _get_embedding_fn():
    if EMBEDDING_PROVIDER == "local":
        return embedding_functions.DefaultEmbeddingFunction()
    return embedding_functions.OpenAIEmbeddingFunction(
        api_key=LLM_API_KEY,
        api_base=LLM_BASE_URL if "api.openai.com" not in LLM_BASE_URL else None,
        model_name=EMBEDDING_MODEL,
    )


def _get_collection(create: bool = False) -> chromadb.Collection:
    client = chromadb.PersistentClient(path=str(VECTOR_STORE_PATH))
    ef = _get_embedding_fn()
    if create:
        # Delete existing collection so re-indexing is always clean
        try:
            client.delete_collection(CHROMA_COLLECTION)
        except Exception:
            pass
        return client.create_collection(
            name=CHROMA_COLLECTION,
            embedding_function=ef,
            metadata={"hnsw:space": "cosine"},
        )
    return client.get_or_create_collection(
        name=CHROMA_COLLECTION,
        embedding_function=ef,
        metadata={"hnsw:space": "cosine"},
    )


# ── Chunking ─────────────────────────────────────────────────────────────────

def _extract_heading(text: str) -> str:
    """Return the last markdown heading found above or within `text`."""
    headings = re.findall(r"^#{1,4}\s+(.+)$", text, re.MULTILINE)
    return headings[-1].strip() if headings else ""


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into token-bounded chunks with overlap."""
    tokens = _enc.encode(text)
    chunks: list[str] = []
    start = 0
    while start < len(tokens):
        end = min(start + chunk_size, len(tokens))
        chunk_tokens = tokens[start:end]
        chunks.append(_enc.decode(chunk_tokens))
        if end == len(tokens):
            break
        start = end - overlap
    return chunks


# ── Document parsing ──────────────────────────────────────────────────────────

def _parse_document(path: Path) -> tuple[dict[str, Any], str]:
    """Return (front_matter_dict, body_text) for a Markdown file."""
    post = frontmatter.load(str(path))
    meta: dict[str, Any] = dict(post.metadata)
    body: str = post.content
    return meta, body


# ── Indexing ──────────────────────────────────────────────────────────────────

def build_index(verbose: bool = True) -> int:
    """
    Index all Markdown files in KNOWLEDGE_BASE_PATH into ChromaDB.

    Returns the total number of chunks indexed.
    """
    kb_path = Path(KNOWLEDGE_BASE_PATH)
    md_files = sorted(kb_path.glob("*.md"))

    if not md_files:
        raise FileNotFoundError(f"No .md files found in {kb_path}")

    collection = _get_collection(create=True)

    ids: list[str] = []
    documents: list[str] = []
    metadatas: list[dict[str, Any]] = []

    for doc_path in md_files:
        meta, body = _parse_document(doc_path)

        # ── Derive metadata ─────────────────────────────────────────────────
        status = str(meta.get("status", "unknown")).lower()
        policy_authority = str(meta.get("policy_authority", "none")).lower()
        audience = str(meta.get("audience", "customer")).lower()
        document_id = str(meta.get("document_id", doc_path.stem))
        title = str(meta.get("title", doc_path.stem))

        chunks = _chunk_text(body, CHUNK_SIZE, CHUNK_OVERLAP)

        for i, chunk in enumerate(chunks):
            heading = _extract_heading(chunk) or title
            chunk_id = f"{doc_path.stem}__chunk_{i}"

            ids.append(chunk_id)
            documents.append(chunk)
            metadatas.append(
                {
                    "source_file": doc_path.name,
                    "document_id": document_id,
                    "title": title,
                    "heading": heading,
                    "status": status,
                    "policy_authority": policy_authority,
                    "audience": audience,
                    "chunk_index": i,
                }
            )

        if verbose:
            print(f"  Indexed {doc_path.name} -> {len(chunks)} chunk(s)")

    # Batch upsert
    collection.add(ids=ids, documents=documents, metadatas=metadatas)

    if verbose:
        print(f"\n[OK] Total chunks indexed: {len(ids)}")

    return len(ids)


# ── Retrieval ─────────────────────────────────────────────────────────────────

class RetrievedChunk:
    """A single passage returned by the retrieval layer."""

    def __init__(
        self,
        text: str,
        source_file: str,
        heading: str,
        status: str,
        policy_authority: str,
        audience: str,
        document_id: str,
        score: float,
    ) -> None:
        self.text = text
        self.source_file = source_file
        self.heading = heading
        self.status = status
        self.policy_authority = policy_authority
        self.audience = audience
        self.document_id = document_id
        self.score = score

    def citation(self) -> str:
        """Human-readable citation string."""
        return f"{self.source_file} § {self.heading}"

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "source_file": self.source_file,
            "heading": self.heading,
            "status": self.status,
            "policy_authority": self.policy_authority,
            "audience": self.audience,
            "document_id": self.document_id,
            "score": self.score,
        }


def _rank_boost(chunk: RetrievedChunk) -> float:
    """
    Return an additive score boost so authoritative active docs float to the top.
    Higher = better.
    """
    boost = 0.0
    if chunk.policy_authority == "official":
        boost += 0.15
    if chunk.status == "active":
        boost += 0.10
    if chunk.status == "superseded":
        boost -= 0.30
    return boost


def retrieve(query: str, k: int = RETRIEVAL_TOP_K) -> list[RetrievedChunk]:
    """
    Retrieve the most relevant chunks for `query`.

    Filtering rules:
      - audience == "internal" chunks are ALWAYS excluded.
      - superseded chunks are deprioritised; included only if nothing better found.

    Returns at most `k` chunks, re-ranked by (cosine_score + boost).
    """
    try:
        collection = _get_collection(create=False)
        total_count = collection.count()
        if total_count == 0:
            return []
    except Exception:
        return []

    fetch_k = min(k * 4, total_count)
    if fetch_k <= 0:
        return []

    # Over-fetch to allow post-filter
    raw = collection.query(
        query_texts=[query],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    chunks: list[RetrievedChunk] = []
    superseded_chunks: list[RetrievedChunk] = []

    docs = raw["documents"][0]
    metas = raw["metadatas"][0]
    distances = raw["distances"][0]

    for doc, meta, dist in zip(docs, metas, distances):
        # ChromaDB cosine distance → similarity (1 = identical, 0 = orthogonal)
        cosine_sim = 1.0 - dist

        audience = str(meta.get("audience", "customer")).lower()
        status = str(meta.get("status", "unknown")).lower()

        # Hard exclude internal documents
        if audience == "internal":
            continue

        chunk = RetrievedChunk(
            text=doc,
            source_file=meta.get("source_file", ""),
            heading=meta.get("heading", ""),
            status=status,
            policy_authority=str(meta.get("policy_authority", "none")).lower(),
            audience=audience,
            document_id=meta.get("document_id", ""),
            score=cosine_sim,
        )

        if status == "superseded":
            superseded_chunks.append(chunk)
        else:
            chunks.append(chunk)

    # Re-rank by adjusted score
    chunks.sort(key=lambda c: c.score + _rank_boost(c), reverse=True)
    result = chunks[:k]

    # If we don't have enough, pad with superseded (clearly marked)
    if len(result) < k:
        superseded_chunks.sort(key=lambda c: c.score, reverse=True)
        result.extend(superseded_chunks[: k - len(result)])

    return result
