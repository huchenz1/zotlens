"""Embedding and reranking backends.

Two independent switches:

  ZOTLENS_EMBED  = local | api     embedding model
  ZOTLENS_RERANK = local | llm | none

`local` uses sentence-transformers on MPS/CUDA/CPU (Qwen3-Embedding-0.6B / Qwen3-Reranker-0.6B).
`api` talks to any OpenAI-compatible /embeddings endpoint; `llm` reranks listwise with the chat
model. With ZOTLENS_EMBED=api and ZOTLENS_RERANK=llm the app needs **no local model at all**,
which is the right preset for a small library or a machine without 2 GB to spare.
"""
from __future__ import annotations

import json
import re
import threading
import time

import numpy as np

from .config import (DATA_DIR, EMBED_API_BASE, EMBED_API_BATCH, EMBED_API_DIM, EMBED_API_KEY, EMBED_API_WORKERS,
                     EMBED_API_MODEL, EMBED_BACKEND, EMBED_MODEL, EMBED_QUERY_PREFIX, RERANK_API_BASE,
                     RERANK_API_KEY, RERANK_API_MODEL, RERANK_BACKEND, RERANK_MODEL)

_lock = threading.RLock()
_embedder = None
_reranker = None
_dim: int | None = None

QUERY_PROMPT = "Instruct: Given a research question, retrieve passages from scientific papers that answer it\nQuery:"
META = DATA_DIR / "embed_meta.json"


def device() -> str:
    import torch
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ---------------------------------------------------------------- local models
def embedder():
    global _embedder
    with _lock:
        if _embedder is None:
            import torch
            from sentence_transformers import SentenceTransformer
            dev = device()
            _embedder = SentenceTransformer(
                str(EMBED_MODEL), device=dev,
                model_kwargs={"torch_dtype": torch.float16 if dev != "cpu" else torch.float32},
                tokenizer_kwargs={"padding_side": "left"},
            )
            _embedder.max_seq_length = 512
    return _embedder


def reranker():
    global _reranker
    with _lock:
        if _reranker is None:
            import torch
            from sentence_transformers import CrossEncoder
            dev = device()
            _reranker = CrossEncoder(
                str(RERANK_MODEL), device=dev, max_length=640,
                model_kwargs={"torch_dtype": torch.float16 if dev != "cpu" else torch.float32},
            )
    return _reranker


# ---------------------------------------------------------------- api embedding
def _api_client():
    import httpx
    from openai import OpenAI
    return OpenAI(base_url=EMBED_API_BASE, api_key=EMBED_API_KEY or "none",
                  http_client=httpx.Client(trust_env=False, timeout=180))


def _api_embed(texts: list[str]) -> np.ndarray:
    """Embed through an OpenAI-compatible endpoint, a few requests in flight at once.

    Indexing is latency-bound (one 64-text request takes ~3.6 s of mostly waiting), so parallel
    batches give a near-linear speed-up until the provider rate-limits us."""
    from concurrent.futures import ThreadPoolExecutor

    batches = [texts[i:i + EMBED_API_BATCH] for i in range(0, len(texts), EMBED_API_BATCH)]
    out: list[list[float] | None] = [None] * len(batches)

    def run(i: int, batch: list[str]) -> None:
        out[i] = _embed_batch(batch)

    if EMBED_API_WORKERS > 1 and len(batches) > 1:
        with ThreadPoolExecutor(max_workers=EMBED_API_WORKERS) as pool:
            list(pool.map(lambda p: run(*p), enumerate(batches)))
    else:
        for i, b in enumerate(batches):
            run(i, b)

    E = np.asarray([v for chunk in out for v in chunk], dtype=np.float32)
    n = np.linalg.norm(E, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return E / n


def _embed_batch(batch: list[str]) -> list[list[float]]:
    """One request, with backoff. Providers rate-limit per token quota (HTTP 429), and a careless
    retry budget kills a 20-minute index build at 90% — so back off generously and honour Retry-After."""
    import random

    client = _api_client()
    extra = {"dimensions": EMBED_API_DIM} if EMBED_API_DIM else {}
    batch = [t.replace("\n", " ")[:8000] or " " for t in batch]
    for attempt in range(8):
        try:
            r = client.embeddings.create(model=EMBED_API_MODEL, input=batch, **extra)
            return [d.embedding for d in sorted(r.data, key=lambda d: d.index)]
        except Exception as exc:  # noqa: BLE001
            if attempt == 7:
                raise
            delay = min(60.0, 2.0 * (2 ** attempt)) + random.uniform(0, 1.5)
            headers = getattr(getattr(exc, "response", None), "headers", {}) or {}
            try:
                delay = max(delay, float(headers.get("retry-after", 0) or 0))
            except (TypeError, ValueError):
                pass
            print(f"    [embed] retry {attempt + 1}/7 after {delay:.0f}s: {type(exc).__name__}", flush=True)
            time.sleep(delay)
    return []


def embed_backend() -> dict:
    """Describe the active embedding backend (also persisted for the index builder)."""
    return {"backend": EMBED_BACKEND,
            "model": EMBED_API_MODEL if EMBED_BACKEND == "api" else EMBED_MODEL.name}


def embed_dim() -> int:
    """Vector dimension of the active backend (probes the API once, then cached on disk)."""
    global _dim
    with _lock:
        if _dim is not None:
            return _dim
        if EMBED_BACKEND == "api":
            if META.exists():
                try:
                    m = json.loads(META.read_text())
                    if m.get("backend") == "api" and m.get("model") == EMBED_API_MODEL and m.get("dim"):
                        _dim = int(m["dim"])
                        return _dim
                except Exception:
                    pass
            _dim = int(_api_embed(["dimension probe"]).shape[1])
        else:
            _dim = int(embedder().get_sentence_embedding_dimension())
        META.parent.mkdir(parents=True, exist_ok=True)
        META.write_text(json.dumps({**embed_backend(), "dim": _dim}, indent=1))
        return _dim


# ---------------------------------------------------------------- public API
def embed_docs(texts: list[str], batch_size: int = 32, show_progress: bool = False) -> np.ndarray:
    if not texts:
        return np.zeros((0, embed_dim()), dtype=np.float16)
    if EMBED_BACKEND == "api":
        return _api_embed(texts).astype(np.float16)
    return embedder().encode(texts, batch_size=batch_size, normalize_embeddings=True,
                             convert_to_numpy=True, show_progress_bar=show_progress).astype(np.float16)


def embed_query(text: str | list[str]) -> np.ndarray:
    single = isinstance(text, str)
    items = [text] if single else list(text)
    if EMBED_BACKEND == "api":
        prefix = EMBED_QUERY_PREFIX
        v = _api_embed([prefix + t for t in items]).astype(np.float32)
    else:
        v = embedder().encode(items, prompt=QUERY_PROMPT, normalize_embeddings=True,
                             convert_to_numpy=True).astype(np.float32)
    return v[0] if single else v


def rerank_kind() -> str:
    return RERANK_BACKEND


def rerank(query: str, docs: list[str], batch_size: int = 16) -> np.ndarray:
    """Scores that are higher-is-better. `local` returns cross-encoder logits (any range),
    `api`/`llm` return 0..1 relevance, `none` returns zeros so the caller keeps the RRF order."""
    if not docs or RERANK_BACKEND == "none":
        return np.zeros(len(docs), dtype=np.float32)
    if RERANK_BACKEND == "llm":
        return _llm_rerank(query, docs)
    if RERANK_BACKEND == "api":
        return _api_rerank(query, docs)
    with _lock:
        scores = reranker().predict([(query, d) for d in docs], batch_size=batch_size,
                                    convert_to_numpy=True, show_progress_bar=False)
    return np.asarray(scores, dtype=np.float32)


def _api_rerank(query: str, docs: list[str]) -> np.ndarray:
    """Rerank through a Cohere/Jina/SiliconFlow-style POST /rerank endpoint."""
    import httpx

    scores = np.zeros(len(docs), dtype=np.float32)
    payload = {"model": RERANK_API_MODEL, "query": query,
               "documents": [" ".join(d.split()[:300]) for d in docs], "top_n": len(docs)}
    headers = {"Authorization": f"Bearer {RERANK_API_KEY}", "Content-Type": "application/json"}
    for attempt in range(3):
        try:
            r = httpx.post(f"{RERANK_API_BASE.rstrip('/')}/rerank", json=payload, headers=headers,
                           timeout=60, trust_env=False)
            r.raise_for_status()
            body = r.json()
            for item in body.get("results", []):
                i = int(item["index"])
                if 0 <= i < len(docs):
                    scores[i] = float(item.get("relevance_score", 0.0))
            mx = float(scores.max())
            if mx > 1.0:                       # some providers return logits instead of 0..1
                scores = scores / mx
            return scores
        except Exception:
            if attempt == 2:
                return np.zeros(len(docs), dtype=np.float32)
            time.sleep(1.0 * (attempt + 1))
    return scores


def _llm_rerank(query: str, docs: list[str]) -> np.ndarray:
    """Listwise rerank with the chat model: one call, returns 0..1 scores by rank position."""
    from .config import llm_config
    from openai import OpenAI
    import httpx

    cfg = llm_config()
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"] or "none",
                    http_client=httpx.Client(trust_env=False, timeout=120))
    listing = "\n".join(f"[{i + 1}] {' '.join(d.split()[:70])}" for i, d in enumerate(docs))
    prompt = (f"Question: {query}\n\nCandidate passages:\n{listing}\n\n"
              f"Rank the passages by how likely they contain the answer to the question. "
              f"Reply with JSON only: {{\"ranking\": [ids, most relevant first]}} using the numbers above.")
    try:
        r = client.chat.completions.create(model=cfg["model"], messages=[{"role": "user", "content": prompt}],
                                           max_tokens=500, temperature=0,
                                           extra_body=cfg["extra_body"] or None)
        raw = r.choices[0].message.content or ""
    except Exception:
        return np.zeros(len(docs), dtype=np.float32)
    ids = [int(x) for x in re.findall(r"\d+", raw)]
    order = []
    for i in ids:
        if 1 <= i <= len(docs) and i not in order:
            order.append(i)
    order += [i for i in range(1, len(docs) + 1) if i not in order]
    scores = np.zeros(len(docs), dtype=np.float32)
    n = max(len(order) - 1, 1)
    for rank, i in enumerate(order):
        scores[i - 1] = 1.0 - rank / n          # 1.0 best … 0.0 worst
    return scores
