"""Per-corpus vector storage with strict physical and logical isolation.

Methodological Safeguards:
1. Physical and logical isolation: Each corpus has its own dedicated directory
   under STORAGE_DIR / <corpus_id>. Cross-corpus retrieval is physically impossible.
2. Similarity is not confidence: Scores are explicitly labeled and recorded as
   mathematical vector proximity metrics (cosine similarity / distance), never as
   confidence, evidential strength, or probability.
3. Embedding model provenance: The exact model name, dimension, and metric are
   stored in index_metadata.json for reproducibility.
"""

from __future__ import annotations

import json
import shutil
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import chromadb
from chromadb.utils import embedding_functions

from src.config import STORAGE_DIR
from src.ingest import DocumentChunk

# Explicitly documented embedding model configuration.
# Model: all-MiniLM-L6-v2 running via local ONNX runtime (384-dimensional).
# Rationale: Operates 100% locally and offline without external API rate limits or latency.
# Methodological Trade-off: High speed and deterministic consistency, but optimized primarily
# for modern English academic text; non-English or archaic orthography may exhibit reduced
# semantic sensitivity, making the transparent lexical scanner a vital complement.
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2 (ONNX)"
EMBEDDING_DIMENSION = 384
DISTANCE_METRIC = "cosine"


@dataclass
class SemanticResult:
    """A retrieved chunk with its geometric proximity metric."""
    chunk: DocumentChunk
    similarity_score: float  # Mathematical cosine similarity: 1.0 - (distance / 2.0)
    distance: float          # Raw distance metric
    rank: int                # 1-indexed rank

    def to_dict(self) -> dict[str, Any]:
        return {
            "chunk": self.chunk.to_dict(),
            "similarity_score": round(self.similarity_score, 4),
            "distance": round(self.distance, 4),
            "rank": self.rank,
            "metric_type": "cosine_similarity",
            "metric_note": (
                "Notice: This score is a mathematical vector proximity metric, "
                "not an indicator of truth, confidence, evidential strength, or probability."
            ),
        }


class CorpusVectorStore:
    """Isolated vector store for a single research corpus."""

    def __init__(self, corpus_id: str):
        self.corpus_id = corpus_id
        self.storage_path: Path = STORAGE_DIR / corpus_id
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file: Path = self.storage_path / "index_metadata.json"

        # Initialize isolated persistent ChromaDB client for this corpus directory only
        self.client = chromadb.PersistentClient(path=str(self.storage_path))
        self.embedding_fn = embedding_functions.DefaultEmbeddingFunction()
        self.collection = self.client.get_or_create_collection(
            name=f"corpus_{corpus_id}",
            embedding_function=self.embedding_fn,
            metadata={"hnsw:space": DISTANCE_METRIC},
        )

    def is_indexed(self) -> bool:
        """Check if this corpus store contains indexed records and metadata."""
        if not self.metadata_file.exists():
            return False
        return self.collection.count() > 0

    def get_metadata(self) -> dict[str, Any] | None:
        """Read reproducible index metadata."""
        if self.metadata_file.exists():
            try:
                return json.loads(self.metadata_file.read_text(encoding="utf-8"))
            except Exception:
                return None
        return None

    def index_chunks(self, chunks: list[DocumentChunk]) -> dict[str, Any]:
        """Index chunks into the isolated collection and persist provenance metadata."""
        if not chunks:
            return {"status": "empty", "chunks_indexed": 0}

        # Clear existing entries in this isolated collection to guarantee pristine re-indexing
        existing_count = self.collection.count()
        if existing_count > 0:
            existing = self.collection.get()
            if existing and existing["ids"]:
                self.collection.delete(ids=existing["ids"])

        ids: list[str] = []
        documents: list[str] = []
        metadatas: list[dict[str, Any]] = []

        for chunk in chunks:
            ids.append(chunk.chunk_id)
            documents.append(chunk.text)
            metadatas.append(
                {
                    "doc_id": chunk.doc_id,
                    "doc_title": chunk.doc_title,
                    "page_number": chunk.page_number,
                    "char_start": chunk.char_start,
                    "char_end": chunk.char_end,
                }
            )

        # Batch add to Chroma
        self.collection.add(ids=ids, documents=documents, metadatas=metadatas)

        metadata_record = {
            "corpus_id": self.corpus_id,
            "embedding_model": EMBEDDING_MODEL_NAME,
            "embedding_dimension": EMBEDDING_DIMENSION,
            "distance_metric": DISTANCE_METRIC,
            "chunks_indexed": len(chunks),
            "indexed_at": datetime.now(timezone.utc).isoformat(),
            "methodological_notice": (
                "Embeddings represent dense geometric vectors. Distance/similarity scores "
                "indicate spatial proximity within the vector space, not evidential certainty."
            ),
        }

        self.metadata_file.write_text(
            json.dumps(metadata_record, indent=2), encoding="utf-8"
        )
        return metadata_record

    def search(self, query: str, top_k: int = 5) -> list[SemanticResult]:
        """Query the vector index for semantically proximate chunks.

        Scores are strictly formatted as similarity/distance metrics.
        """
        count = self.collection.count()
        if count == 0:
            return []

        actual_k = min(top_k, count)
        results = self.collection.query(
            query_texts=[query],
            n_results=actual_k,
            include=["documents", "metadatas", "distances"],
        )

        semantic_results: list[SemanticResult] = []

        if not results or not results["ids"] or not results["ids"][0]:
            return semantic_results

        ids = results["ids"][0]
        docs = results["documents"][0] if results["documents"] else []
        metas = results["metadatas"][0] if results["metadatas"] else []
        distances = results["distances"][0] if results["distances"] else []

        for rank_idx, chunk_id in enumerate(ids):
            raw_dist = float(distances[rank_idx]) if rank_idx < len(distances) else 0.0
            meta = metas[rank_idx] if rank_idx < len(metas) else {}
            text = docs[rank_idx] if rank_idx < len(docs) else ""

            # In cosine space, distance is 1 - cos(theta) in [0, 2].
            # Cosine similarity is 1.0 - raw_dist, bounded to [-1.0, 1.0].
            # We map to [0.0, 1.0] for standard display as (1.0 - raw_dist / 2.0)
            similarity = max(0.0, min(1.0, 1.0 - (raw_dist / 2.0)))

            chunk = DocumentChunk(
                chunk_id=chunk_id,
                doc_id=meta.get("doc_id", ""),
                doc_title=meta.get("doc_title", ""),
                page_number=int(meta.get("page_number", 1)),
                char_start=int(meta.get("char_start", 0)),
                char_end=int(meta.get("char_end", 0)),
                text=text,
            )

            semantic_results.append(
                SemanticResult(
                    chunk=chunk,
                    similarity_score=similarity,
                    distance=raw_dist,
                    rank=rank_idx + 1,
                )
            )

        return semantic_results

    def clear(self) -> None:
        """Completely purge this corpus index and its storage directory safely on Windows."""
        if self.storage_path.exists():
            for item in self.storage_path.iterdir():
                try:
                    if item.is_file() or item.is_symlink():
                        item.unlink(missing_ok=True)
                    elif item.is_dir():
                        shutil.rmtree(item, ignore_errors=True)
                except Exception:
                    pass
        self.storage_path.mkdir(parents=True, exist_ok=True)
