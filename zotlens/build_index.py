"""Build / incrementally update the ZotLens index.

Stages
  1. metadata snapshot  -> data/papers.json, data/collections.json
  2. chunking + BM25    -> data/index.sqlite (chunks table + FTS5, papers FTS5)
  3. paper embeddings   -> data/paper_emb.npy          (title + abstract, fast)
  4. chunk embeddings   -> data/chunk_emb.npy + chunk_emb_ids.npy (slow, checkpointed,
                           re-uses vectors of unchanged chunks via sha1)

Usage:  python -m zotlens.build_index [--no-chunk-embed]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import time

import numpy as np

from .config import CHUNK_OVERLAP, CHUNK_WORDS, DATA_DIR, MAX_CHUNKS_PER_PAPER, ZOTERO_API
from .zotero_data import chunk_text, load_library, read_paper_text

IDX = DATA_DIR / "index.sqlite"
PAPERS = DATA_DIR / "papers.json"
PAPER_EMB = DATA_DIR / "paper_emb.npy"
CHUNK_EMB = DATA_DIR / "chunk_emb.npy"
CHUNK_IDS = DATA_DIR / "chunk_emb_ids.npy"
STATUS = DATA_DIR / "status.json"
VECTORS_SIG = DATA_DIR / "vectors_model.json"   # which model actually produced the .npy vectors


def sha(s: str) -> str:
    return hashlib.sha1(s.encode("utf-8", "ignore")).hexdigest()


def _read_sig() -> dict:
    if not VECTORS_SIG.exists():
        return {}
    try:
        return json.loads(VECTORS_SIG.read_text())
    except Exception:
        return {}


def zotero_version() -> int | None:
    """Zotero's library version: any add/edit/delete bumps it. Used to detect a stale index
    (item *counts* are not comparable: top-level items include notes and webpages)."""
    try:
        import httpx
        r = httpx.get(f"{ZOTERO_API}/users/0/items/top", params={"limit": 1}, timeout=3, trust_env=False)
        return int(r.headers.get("Last-Modified-Version") or 0) or None
    except Exception:
        return None


def paper_doc(p: dict) -> str:
    auth = ", ".join(p["authors"][:6])
    return f"Title: {p['title']}\nAuthors: {auth}\nAbstract: {p['abstract'] or ''}"


def save_status(**kw):
    s = json.loads(STATUS.read_text()) if STATUS.exists() else {}
    s.update(kw, updated=time.strftime("%Y-%m-%d %H:%M:%S"))
    STATUS.write_text(json.dumps(s, indent=1))


def atomic_save(path, arr):
    tmp = path.with_suffix(".tmp.npy")
    np.save(tmp, arr)
    os.replace(tmp, path)


def load_old_vectors(dim: int, sig: dict) -> tuple[dict, dict]:
    """sha -> vector for previously embedded chunks / papers (incremental rebuilds).

    Reuse is only valid for vectors produced by the *same* embedding model: a different backend or
    model means a different vector space, even when the dimension happens to match (e.g. Qwen3-0.6B
    and bge-m3 are both 1024-d). The signature is recorded in vectors_model.json and, crucially, the
    stale .npy files are deleted *before* a rebuild starts — otherwise an interrupted run would leave
    old vectors on disk with a new signature and silently mix two vector spaces.
    """
    chunk_vec, paper_vec = {}, {}
    prev = _read_sig()
    if not prev or prev.get("backend") != sig["backend"] or prev.get("model") != sig["model"] or prev.get("dim") != dim:
        print(f"    embedding model changed ({prev.get('model', 'none')} -> {sig['model']}): old vectors deleted")
        for f in (CHUNK_EMB, CHUNK_IDS, PAPER_EMB):
            f.unlink(missing_ok=True)
        VECTORS_SIG.write_text(json.dumps({**sig, "dim": dim, "started": time.strftime("%Y-%m-%d %H:%M:%S")}))
        return {}, {}
    if IDX.exists() and CHUNK_EMB.exists() and CHUNK_IDS.exists():
        con = sqlite3.connect(IDX)
        id2sha = dict(con.execute("SELECT id, sha FROM chunks"))
        con.close()
        E, ids = np.load(CHUNK_EMB), np.load(CHUNK_IDS)
        if E.ndim == 2 and E.shape[1] == dim:
            for i, cid in enumerate(ids):
                if int(cid) in id2sha:
                    chunk_vec[id2sha[int(cid)]] = E[i]
        else:
            print(f"    (old chunk vectors have dim {E.shape[-1]}, current backend is {dim} -> full rebuild)")
    if PAPERS.exists() and PAPER_EMB.exists():
        old = json.loads(PAPERS.read_text())
        E = np.load(PAPER_EMB)
        if E.ndim == 2 and E.shape[1] == dim:
            for i, p in enumerate(old):
                if i < len(E):
                    paper_vec[sha(paper_doc(p))] = E[i]
    return chunk_vec, paper_vec


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-chunk-embed", action="store_true")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--limit", type=int, default=0, help="only the N most recent papers (debug / quick test)")
    args = ap.parse_args(argv)
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    t0 = time.time()

    from . import models
    dim = models.embed_dim()
    sig = models.embed_backend()
    print(f"[0] embedding backend: {sig} dim={dim}", flush=True)
    old_chunk_vec, old_paper_vec = load_old_vectors(dim, sig)
    print(f"    reusable vectors: {len(old_chunk_vec)} chunks, {len(old_paper_vec)} papers", flush=True)

    # 1. metadata
    lib = load_library()
    papers = lib["papers"]
    papers.sort(key=lambda p: p["dateAdded"], reverse=True)
    if args.limit:
        papers = papers[: args.limit]
    PAPERS.write_text(json.dumps(papers, ensure_ascii=False))
    (DATA_DIR / "collections.json").write_text(json.dumps(lib["collections"], ensure_ascii=False))
    print(f"[1] {len(papers)} papers ({time.time()-t0:.1f}s)", flush=True)

    # 2. chunks + FTS
    tmp_idx = IDX.with_suffix(".building.sqlite")
    if tmp_idx.exists():
        tmp_idx.unlink()
    con = sqlite3.connect(tmp_idx)
    con.executescript("""
        CREATE TABLE chunks(id INTEGER PRIMARY KEY, paper INTEGER, ord INTEGER, text TEXT, sha TEXT);
        CREATE INDEX chunks_paper ON chunks(paper);
        CREATE VIRTUAL TABLE chunks_fts USING fts5(text, content='chunks', content_rowid='id',
                                                   tokenize='porter unicode61');
        CREATE VIRTUAL TABLE papers_fts USING fts5(title, authors, abstract, tokenize='porter unicode61');
    """)
    rows, cid = [], 0
    for pi, p in enumerate(papers):
        con.execute("INSERT INTO papers_fts(rowid, title, authors, abstract) VALUES (?,?,?,?)",
                    (pi, p["title"], " ".join(p["authors"]), p["abstract"]))
        body = read_paper_text(p)
        chunks = chunk_text(body, CHUNK_WORDS, CHUNK_OVERLAP, MAX_CHUNKS_PER_PAPER)
        if not chunks and p["abstract"]:
            chunks = [p["abstract"]]
        for k, c in enumerate(chunks):
            rows.append((cid, pi, k, c, sha(p["title"] + "\n" + c)))
            cid += 1
    con.executemany("INSERT INTO chunks VALUES (?,?,?,?,?)", rows)
    con.execute("INSERT INTO chunks_fts(chunks_fts) VALUES ('rebuild')")
    con.commit()
    con.close()
    os.replace(tmp_idx, IDX)
    print(f"[2] {len(rows)} chunks + FTS5 ({time.time()-t0:.1f}s)", flush=True)
    save_status(papers=len(papers), chunks=len(rows), chunks_embedded=0, stage="chunks indexed",
                embed=models.embed_backend(), dim=dim, zotero_version=zotero_version())

    # 3. paper embeddings
    docs = [paper_doc(p) for p in papers]
    E = np.zeros((len(docs), dim), dtype=np.float16)
    todo = []
    for i, d in enumerate(docs):
        v = old_paper_vec.get(sha(d))
        if v is not None:
            E[i] = v
        else:
            todo.append(i)
    if todo:
        E[todo] = models.embed_docs([docs[i] for i in todo], batch_size=args.batch, show_progress=True)
    atomic_save(PAPER_EMB, E)
    print(f"[3] paper embeddings: {len(todo)} new, {len(docs)-len(todo)} reused ({time.time()-t0:.1f}s)", flush=True)
    save_status(stage="paper embeddings ready")

    # 4. chunk embeddings (checkpointed)
    titles = [p["title"] for p in papers]
    done_ids, done_vecs, todo = [], [], []
    for (cid, pi, k, text, h) in rows:
        v = old_chunk_vec.get(h)
        if v is not None:
            done_ids.append(cid)
            done_vecs.append(v)
        else:
            todo.append((cid, f"Title: {titles[pi]}\n{text}"))
    print(f"[4] chunk embeddings: {len(done_ids)} reused, {len(todo)} to embed", flush=True)

    def flush():
        if done_ids:
            atomic_save(CHUNK_EMB, np.stack(done_vecs).astype(np.float16))
            atomic_save(CHUNK_IDS, np.asarray(done_ids, dtype=np.int64))
        save_status(chunks_embedded=len(done_ids), stage="embedding chunks" if todo else "done",
                    embed=models.embed_backend(), dim=dim)

    flush()
    if args.no_chunk_embed:
        return
    BLOCK = 512
    t1 = time.time()
    for s in range(0, len(todo), BLOCK):
        block = todo[s:s + BLOCK]
        V = models.embed_docs([t for _, t in block], batch_size=args.batch)
        done_ids.extend(c for c, _ in block)
        done_vecs.extend(V)
        rate = (s + len(block)) / (time.time() - t1)
        eta = (len(todo) - s - len(block)) / max(rate, 1e-6)
        print(f"    {s+len(block)}/{len(todo)}  {rate:.1f}/s  eta {eta/60:.1f} min", flush=True)
        if (s // BLOCK) % 4 == 3 or s + BLOCK >= len(todo):
            flush()
    todo = []
    flush()
    print(f"[done] total {time.time()-t0:.1f}s", flush=True)


if __name__ == "__main__":
    sys.exit(main())
