"""Query the FAISS/Chroma knowledge stores under `data/*_dbs/` for retrieval-augmented chat.

Each store (e.g. `data/data_science_dbs/`) holds a `faiss_index/` and a `chroma_db/`, built by that
domain's `dataset.py`. `faiss_index/embedder/` carries a `vocab.json` + `projection.npy` pair that
reproduces the log-scaled TF + random-projection embedding the store's vectors were built with; this
module re-implements that same encode step so a query lands in the same space as the indexed
documents, without importing the `data.*_dbs.dataset` modules those stores are built from.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _patch_chromadb_otel() -> None:
    """Stub out the OTLP exporter chromadb imports but never instantiates at default telemetry settings."""
    target = "opentelemetry.exporter.otlp.proto.grpc.trace_exporter"
    if target in sys.modules:
        return
    mod = types.ModuleType(target)

    class OTLPSpanExporter:
        def __init__(self, *args: Any, **kwargs: Any) -> None:
            pass

    mod.OTLPSpanExporter = OTLPSpanExporter  # type: ignore[attr-defined]
    sys.modules[target] = mod


_patch_chromadb_otel()

import chromadb  # noqa: E402
import faiss  # noqa: E402
from chromadb.config import Settings  # noqa: E402

# Auxiliary collections that live alongside the knowledge collection in the same chroma_db but
# are not part of it (a generic empty "default" collection, and a web-search response cache).
_IGNORED_COLLECTIONS = {"default", "web_search_cache"}

_TOKEN_RE = re.compile(r"[a-z]+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


@dataclass
class _Embedder:
    """Mirrors `SyntheticEmbedder.encode` from `data/*_dbs/dataset.py`, reconstructed from its saved state."""

    vocab: dict[str, int]
    projection: np.ndarray

    @classmethod
    def load(cls, embedder_dir: Path) -> "_Embedder":
        vocab = json.loads((embedder_dir / "vocab.json").read_text())
        projection = np.load(embedder_dir / "projection.npy")
        return cls(vocab=vocab, projection=projection)

    def encode(self, text: str) -> np.ndarray:
        token_counts: dict[int, float] = {}
        for token in _tokenize(text):
            idx = self.vocab.get(token)
            if idx is not None:
                token_counts[idx] = token_counts.get(idx, 0.0) + 1.0

        dim = self.projection.shape[1]
        if not token_counts:
            h = int(hashlib.sha256(text.encode()).hexdigest(), 16)
            vector = np.random.default_rng(h).standard_normal(dim).astype(np.float32)
        else:
            indices = np.array(list(token_counts.keys()), dtype=np.int32)
            weights = 1.0 + np.log(np.array(list(token_counts.values()), dtype=np.float32))
            vector = weights @ self.projection[indices]

        norm = np.linalg.norm(vector)
        if norm > 0:
            vector /= norm
        return vector.astype(np.float32)


@dataclass
class KnowledgeStore:
    """A loaded FAISS index paired with the Chroma collection holding its documents' text."""

    name: str
    embedder: _Embedder
    index: Any
    ids: list[str]
    collection: Any

    def search(self, query: str, top_k: int = 4) -> list[dict[str, Any]]:
        """Return the `top_k` documents whose stored vectors are closest to `query`, best first."""
        if not self.ids:
            return []
        vector = self.embedder.encode(query).reshape(1, -1)
        scores, positions = self.index.search(vector, min(top_k, len(self.ids)))
        doc_ids = [self.ids[i] for i in positions[0] if i != -1]
        if not doc_ids:
            return []

        got = self.collection.get(ids=doc_ids, include=["documents", "metadatas"])
        by_id = dict(zip(got["ids"], zip(got["documents"], got["metadatas"], strict=True), strict=True))

        results = []
        for doc_id, score in zip(doc_ids, scores[0], strict=True):
            if doc_id not in by_id:
                continue
            text, metadata = by_id[doc_id]
            results.append({"id": doc_id, "text": text, "metadata": metadata or {}, "score": float(score)})
        return results


def _pick_collection_name(client: Any) -> str:
    names = [c.name for c in client.list_collections()]
    candidates = [n for n in names if n not in _IGNORED_COLLECTIONS]
    if len(candidates) != 1:
        raise RuntimeError(
            f"Expected exactly one knowledge collection in the Chroma DB, found {candidates} (all: {names})"
        )
    return candidates[0]


def load_knowledge_store(db_path: Path) -> KnowledgeStore:
    """Load the FAISS index and its matching Chroma collection from `db_path` (e.g. `data/data_science_dbs`)."""
    faiss_dir = db_path / "faiss_index"
    chroma_dir = db_path / "chroma_db"
    if not faiss_dir.is_dir() or not chroma_dir.is_dir():
        raise FileNotFoundError(f"{db_path} must contain both a faiss_index/ and a chroma_db/ directory")

    index = faiss.read_index(str(faiss_dir / "index.faiss"))
    ids: list[str] = json.loads((faiss_dir / "ids.json").read_text())
    embedder = _Embedder.load(faiss_dir / "embedder")

    client = chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))
    collection = client.get_collection(name=_pick_collection_name(client))

    return KnowledgeStore(name=db_path.name, embedder=embedder, index=index, ids=ids, collection=collection)
