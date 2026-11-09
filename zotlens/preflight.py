"""Preflight: check that the configured embedding / rerank / LLM backends actually work.

    ./run.sh check                 # uses the current env
    ZOTLENS_RERANK=api ... ./run.sh check

It answers, in one shot: is the endpoint reachable, is the API key valid and funded,
what embedding dimension does it return, and how fast is one rerank call. Nothing is
written to the index, so it is safe to run before a full rebuild.
"""
from __future__ import annotations

import sys
import time

import numpy as np

from .config import (EMBED_API_BASE, EMBED_API_MODEL, EMBED_API_KEY, EMBED_BACKEND, EMBED_MODEL,
                     RERANK_API_BASE, RERANK_API_KEY, RERANK_API_MODEL, RERANK_BACKEND, RERANK_MODEL,
                     llm_config)

SAMPLE = [
    "Masked diffusion models decode several tokens in parallel in a single step.",
    "We benchmark GPU kernels for dense matrix multiplication on A100.",
    "Parallel sampling breaks token dependencies and degrades generation quality.",
]
QUERY = "why does parallel sampling hurt quality in masked diffusion models"


def _row(name, ok, detail):
    print(f"  {'PASS' if ok else 'FAIL'}  {name:<26} {detail}")
    return ok


def check_embed() -> bool:
    from . import models
    label = f"{EMBED_BACKEND}:{EMBED_API_MODEL if EMBED_BACKEND == 'api' else EMBED_MODEL.name}"
    try:
        t = time.time()
        v = models.embed_docs(SAMPLE)
        dt = time.time() - t
        q = models.embed_query(QUERY)
        sim = float(np.dot(v[2].astype(np.float32), q))
        return _row("embedding", v.shape[1] == q.shape[0],
                    f"{label}  dim={v.shape[1]}  {dt:.2f}s/3 texts  "
                    f"sim(correct doc)={sim:.3f}  (预期高于另外两条)")
    except Exception as e:  # noqa: BLE001
        return _row("embedding", False, f"{label}  {type(e).__name__}: {str(e)[:120]}")


def check_rerank() -> bool:
    from . import models
    label = (f"{RERANK_BACKEND}:{RERANK_API_MODEL}" if RERANK_BACKEND == "api" else
             f"{RERANK_BACKEND}:{RERANK_MODEL.name if RERANK_BACKEND == 'local' else RERANK_BACKEND}")
    try:
        t = time.time()
        s = models.rerank(QUERY, SAMPLE)
        dt = time.time() - t
        order = np.argsort(-s)
        ok = bool(len(s) == len(SAMPLE) and order[0] == 2)
        return _row("rerank", ok, f"{label}  {dt:.2f}s/3 docs  order={list(order)}  "
                                  f"scores={[round(float(x), 3) for x in s]}")
    except Exception as e:  # noqa: BLE001
        return _row("rerank", False, f"{label}  {type(e).__name__}: {str(e)[:120]}")


def check_llm() -> bool:
    cfg = llm_config()
    label = f"{cfg['backend']}:{cfg['model'] or 'auto'}"
    try:
        import httpx
        from openai import OpenAI
        c = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"] or "none",
                   http_client=httpx.Client(trust_env=False, timeout=30))
        t = time.time()
        r = c.chat.completions.create(model=cfg["model"], messages=[{"role": "user", "content": "回复：ok"}],
                                      max_tokens=8, extra_body=cfg["extra_body"] or None)
        return _row("chat LLM", True, f"{label}  {time.time()-t:.2f}s  reply={r.choices[0].message.content!r}")
    except Exception as e:  # noqa: BLE001
        return _row("chat LLM", False, f"{label}  {type(e).__name__}: {str(e)[:120]}")


def main() -> int:
    print("ZotLens preflight\n" + "=" * 78)
    print(f"embedding endpoint : {EMBED_API_BASE if EMBED_BACKEND == 'api' else EMBED_MODEL}")
    print(f"rerank    endpoint : {RERANK_API_BASE if RERANK_BACKEND == 'api' else RERANK_BACKEND}")
    print(f"keys set           : embed={bool(EMBED_API_KEY)} rerank={bool(RERANK_API_KEY)}")
    print("-" * 78)
    results = [check_embed(), check_rerank(), check_llm()]
    print("-" * 78)
    bad = results.count(False)
    print("all good" if not bad else f"{bad} check(s) failed — see FAIL lines above")
    if EMBED_BACKEND == "api" or RERANK_BACKEND == "api":
        print("\nnote: switching the embedding backend means the index must be rebuilt (./run.sh index).")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
