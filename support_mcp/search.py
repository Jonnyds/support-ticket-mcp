"""Semantic search over subject+body using OpenAI embeddings (cached to disk).
Needs OPENAI_API_KEY; the SQL tools work without it."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import db

EMBED_MODEL = os.getenv("OPENAI_EMBED_MODEL", "text-embedding-3-small")
CACHE_DIR = Path(__file__).resolve().parent.parent / "data" / ".cache"
MIN_SIMILARITY = float(os.getenv("SEARCH_MIN_SIMILARITY", "0.30"))  # relevance floor (0-1)
MAX_RESULTS = 20      # ceiling on k
MIN_QUERY_LEN = 2     # reject empty / single-char queries
MAX_QUERY_LEN = 1000  # trim pasted walls of text
_BATCH = 256          # texts per embedding API call

_VECTORS: np.ndarray | None = None
_META: list[dict] | None = None
_TEXTS: list[str] | None = None


def _cell(df, col, i):
    """Return the cell value as a string, or None if missing/blank."""
    if col not in df.columns:
        return None
    v = df[col].iloc[i]
    return None if pd.isna(v) else str(v)


def _build_texts() -> tuple[list[str], list[dict]]:
    """Build the embed-texts (subject+body) and per-ticket metadata lists."""
    df = db.get_dataframe()
    texts, meta = [], []
    for i in range(len(df)):
        subject = _cell(df, "subject", i) or ""
        body = _cell(df, "body", i) or ""
        # embed subject+body only, answer stays in metadata not the vector
        texts.append((subject + "\n" + body).strip())
        meta.append({
            "row_id": i,
            "subject": subject,
            "answer": (_cell(df, "answer", i) or "")[:500],
            "type": _cell(df, "type", i),
            "queue": _cell(df, "queue", i),
            "priority": _cell(df, "priority", i),
            "language": _cell(df, "language", i),
        })
    return texts, meta


def _cache_key(texts: list[str]) -> str:
    """Short fingerprint of model + row count + sample text. if any change,
    the key changes so a fresh cache is built instead of loading stale vectors."""
    h = hashlib.sha256(EMBED_MODEL.encode())
    h.update(str(len(texts)).encode())
    h.update("".join(texts[:50]).encode())
    return h.hexdigest()[:16]


def _embed_batch(client, batch: list[str]) -> list[list[float]]:
    """Embed one batch of texts via OpenAI, return their vectors."""
    resp = client.embeddings.create(model=EMBED_MODEL, input=batch)
    return [d.embedding for d in resp.data]


def _ensure_index() -> None:
    """Load the cached index, or build and cache it on first search."""
    global _VECTORS, _META, _TEXTS
    if _VECTORS is not None:
        return

    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError(
            "search_tickets needs OPENAI_API_KEY. The SQL tools work without it."
        )

    texts, meta = _build_texts()
    key = _cache_key(texts)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    vec_path = CACHE_DIR / f"emb_{key}.npy"
    meta_path = CACHE_DIR / f"meta_{key}.json"

    if vec_path.exists() and meta_path.exists():
        _VECTORS = np.load(vec_path)
        _META = json.loads(meta_path.read_text())
        _TEXTS = texts
        return

    # imported here, not at top, so the SQL tools don't require the openai package
    from openai import OpenAI
    client = OpenAI()
    vectors: list[list[float]] = []
    for start in range(0, len(texts), _BATCH):
        # OpenAI rejects empty input, so swap any blank text for a single space
        batch = [t if t else " " for t in texts[start:start + _BATCH]]
        vectors.extend(_embed_batch(client, batch))

    arr = np.array(vectors, dtype=np.float32)
    # normalize each vector to unit length so cosine similarity == a plain dot product
    arr /= (np.linalg.norm(arr, axis=1, keepdims=True) + 1e-9)
    np.save(vec_path, arr)
    meta_path.write_text(json.dumps(meta))
    _VECTORS, _META, _TEXTS = arr, meta, texts


def search(query: str, k: int = 5) -> dict:
    query = (query or "").strip()
    if len(query) < MIN_QUERY_LEN:
        return {"error": "Query is too short."}
    if len(query) > MAX_QUERY_LEN:
        query = query[:MAX_QUERY_LEN]

    _ensure_index()
    # lazy import (see _ensure_index): keeps openai optional for SQL-only use
    from openai import OpenAI
    client = OpenAI()
    q = client.embeddings.create(model=EMBED_MODEL, input=[query]).data[0].embedding
    qv = np.array(q, dtype=np.float32)
    qv /= (np.linalg.norm(qv) + 1e-9)  # normalize the query the same way

    sims = _VECTORS @ qv  # dot product vs every ticket vector = similarity scores
    k = max(1, min(int(k), MAX_RESULTS))
    top = np.argsort(-sims)[:k]  # indices of the k highest scores (negate = descending)

    # keep only results above the relevance floor (per-result, not all-or-nothing)
    results = []
    for idx in top:
        score = float(sims[idx])
        if score < MIN_SIMILARITY:
            continue
        m = dict(_META[idx])
        m["similarity"] = round(score, 4)
        m["description"] = _TEXTS[idx][:600]
        results.append(m)

    if not results:
        return {"query": query, "results": [],
                "note": "No tickets matched closely enough; the question may be "
                        "out of scope for this dataset."}
    return {"query": query, "results": results}
