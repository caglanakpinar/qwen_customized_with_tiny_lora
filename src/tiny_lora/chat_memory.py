"""Persist chat running-summaries into FAISS and ChromaDB, embedded with the chat model itself.

Every `WRITE_EVERY_N_SUMMARIES`-th time `chat.py` folds older turns into `running_summary`, the
resulting text is embedded with the same Qwen model doing the chatting (mean-pooled last hidden
state, L2-normalized) and appended to a FAISS index and a Chroma collection under `memory_dir`, so
retrieval-time similarity search stays consistent with the model that generated the summaries.

FAISS holds only vectors and ids (`IndexFlatIP`, since normalized vectors make inner product
equal cosine similarity); Chroma is the source of truth for the summary text and metadata, mirroring
the split already used by the offline knowledge stores under `data/*_dbs/dataset.py`.
"""

from __future__ import annotations

import json
import os
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch

# torch and faiss-cpu each bundle their own OpenMP runtime; loading both in one process aborts
# on macOS ("OMP: Error #15") unless this is set before faiss is imported.
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

WRITE_EVERY_N_SUMMARIES = 5

FAISS_SUBDIR = "faiss_index"
CHROMA_SUBDIR = "chroma_db"
COLLECTION = "chat_summaries"


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


@torch.no_grad()
def embed_text(model, tokenizer, text: str) -> np.ndarray:
    """Mean-pool `model`'s last hidden state over `text`'s tokens into one L2-normalized vector."""
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=tokenizer.model_max_length).to(
        model.device
    )
    outputs = model(**inputs, output_hidden_states=True)
    last_hidden = outputs.hidden_states[-1].float()  # (1, seq_len, hidden)
    mask = inputs["attention_mask"].unsqueeze(-1).float()  # (1, seq_len, 1)
    pooled = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)
    vector = pooled.squeeze(0).cpu().numpy().astype(np.float32)
    norm = np.linalg.norm(vector)
    if norm > 0:
        vector /= norm
    return vector


def _append_to_faiss(vector: np.ndarray, doc_id: str, faiss_dir: Path) -> None:
    faiss_dir.mkdir(parents=True, exist_ok=True)
    index_path = faiss_dir / "index.faiss"
    ids_path = faiss_dir / "ids.json"

    if index_path.exists() and ids_path.exists():
        index = faiss.read_index(str(index_path))
        ids: list[str] = json.loads(ids_path.read_text())
    else:
        index = faiss.IndexFlatIP(vector.shape[0])
        ids = []

    if doc_id in ids:
        return
    index.add(vector.reshape(1, -1))
    ids.append(doc_id)

    faiss.write_index(index, str(index_path))
    ids_path.write_text(json.dumps(ids, indent=2))


def _upsert_to_chroma(
    vector: np.ndarray, doc_id: str, text: str, metadata: dict[str, Any], chroma_dir: Path
) -> None:
    chroma_dir.mkdir(parents=True, exist_ok=True)
    client = chromadb.PersistentClient(path=str(chroma_dir), settings=Settings(anonymized_telemetry=False))
    collection = client.get_or_create_collection(name=COLLECTION)
    collection.upsert(
        ids=[doc_id],
        documents=[text],
        metadatas=[metadata],
        embeddings=[vector.tolist()],
    )


def record_summary(
    model,
    tokenizer,
    summary_text: str,
    summary_event_index: int,
    memory_dir: Path,
    source: str,
) -> None:
    """Embed `summary_text` with `model`/`tokenizer` and append it to the FAISS index and Chroma collection."""
    vector = embed_text(model, tokenizer, summary_text)
    doc_id = f"chat-summary-{source}-{summary_event_index}"
    metadata = {"source": source, "summary_event_index": summary_event_index}

    _append_to_faiss(vector, doc_id, memory_dir / FAISS_SUBDIR)
    _upsert_to_chroma(vector, doc_id, summary_text, metadata, memory_dir / CHROMA_SUBDIR)
