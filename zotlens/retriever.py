"""Hybrid retrieval over the local Zotero library.

Pipeline (per query):
  dense(chunks) + dense(papers) + BM25(chunks) + BM25(papers)
    -> Reciprocal Rank Fusion at paper level
    -> cross-encoder rerank of the best passages of the top papers
    -> paper ranking = best passage score; evidence = top passages (+ PDF page)
"""
from __future__ import annotations

import json
import os
import re
import sqlite3
import threading
import time
from functools import lru_cache
from pathlib import Path

import numpy as np

from . import models
from .build_index import CHUNK_EMB, CHUNK_IDS, IDX, PAPER_EMB, PAPERS, STATUS, VECTORS_SIG
from .config import DATA_DIR

STOP = set("""a an the of for to in on and or with without by from is are was were be been being as at
this that these those it its into via vs versus how what why which who whom when where do does did can
could should would may might will shall than then there their them they we our you your i me my about
paper papers work works method methods approach approaches study studies any some find show me please
using use used based between across over under more most less least other others such""".split())

RRF_K = 60
# BM25 is a weak list on this library (0.00 recall@10 for Chinese queries), so it is fused with a
# lower weight than the dense lists; equal weights measurably hurt (see data/eval_results.md).
BM25_WEIGHT = float(os.environ.get("ZOTLENS_BM25_WEIGHT", 1.0))


def fts_query(q: str) -> str | None:
    toks = [t.lower() for t in re.findall(r"[A-Za-z][A-Za-z0-9\-]*|\d+[A-Za-z]+", q)]
    toks = [t.strip("-") for t in toks if t not in STOP and len(t) > 1]
    toks = list(dict.fromkeys(t for t in toks if t))
    if not toks:
        return None
    return " OR ".join(f'"{t}"' for t in toks[:16])


def _norm(s: str) -> str:
    s = re.sub(r"-\s*\n\s*", "", s.lower())
    return re.sub(r"[^a-z0-9]+", " ", s).strip()


@lru_cache(maxsize=48)
def _pdf_pages(path: str) -> tuple[str, ...]:
    import pymupdf
    try:
        with pymupdf.open(path) as doc:
            return tuple(_norm(pg.get_text()) for pg in doc)
    except Exception:
        return tuple()


def locate_page(pdf_path: str | None, passage: str) -> int | None:
    """Find the 1-based PDF page containing a passage taken from Zotero's text cache."""
    if not pdf_path or not Path(pdf_path).exists():
        return None
    pages = _pdf_pages(pdf_path)
    if not pages:
        return None
    w = _norm(passage).split()
    if len(w) < 8:
        return None
    for start in (len(w) // 3, len(w) // 2, 5, 2 * len(w) // 3, 0):
        probe = " ".join(w[start:start + 7])
        hits = [i for i, pg in enumerate(pages) if probe in pg]
        if len(hits) == 1:
            return hits[0] + 1
    # fallback: page with largest 4-gram overlap
    grams = {" ".join(w[i:i + 4]) for i in range(0, len(w) - 4, 2)}
    best, best_i = 0, None
    for i, pg in enumerate(pages):
        s = sum(g in pg for g in grams)
        if s > best:
            best, best_i = s, i
    return best_i + 1 if best_i is not None and best >= 3 else None


def zotero_links(paper: dict, page: int | None = None) -> dict:
    links = {"select": f"zotero://select/library/items/{paper['key']}"}
    if paper.get("pdf_key") and paper.get("pdf_path"):
        links["open_pdf"] = f"zotero://open-pdf/library/items/{paper['pdf_key']}" + (f"?page={page}" if page else "")
    return links


class Library:
    def __init__(self):
        self._lock = threading.RLock()
        self._chunk_mtime = None
        self.dim_warning = None
        self.load()

    # ---------------------------------------------------------------- loading
    def load(self):
        with self._lock:
            self.papers: list[dict] = json.loads(PAPERS.read_text())
            self.key2idx = {}
            for i, p in enumerate(self.papers):
                self.key2idx[p["key"]] = i
                for d in p.get("dup_keys", []):
                    self.key2idx[d] = i
            self.collections = json.loads((DATA_DIR / "collections.json").read_text())
            self.db = sqlite3.connect(f"file:{IDX}?mode=ro", uri=True, check_same_thread=False)
            cp = self.db.execute("SELECT id, paper FROM chunks ORDER BY id").fetchall()
            self.chunk_paper = np.array([p for _, p in cp], dtype=np.int32)
            self.paper_emb = np.load(PAPER_EMB).astype(np.float32) if PAPER_EMB.exists() else None
            self.years = np.array([int(p["year"]) if p["year"] else 0 for p in self.papers])
            self._load_chunk_emb()

    def _load_chunk_emb(self):
        if CHUNK_EMB.exists() and CHUNK_IDS.exists():
            m = CHUNK_EMB.stat().st_mtime
            if m != self._chunk_mtime:
                self.chunk_emb = np.load(CHUNK_EMB).astype(np.float32)
                self.chunk_emb_ids = np.load(CHUNK_IDS)
                self._chunk_mtime = m
        else:
            self.chunk_emb, self.chunk_emb_ids = None, None

    def maybe_reload(self):
        with self._lock:
            try:
                self._load_chunk_emb()
            except Exception:
                pass

    def status(self) -> dict:
        s = json.loads(STATUS.read_text()) if STATUS.exists() else {}
        s["chunks_embedded_loaded"] = 0 if self.chunk_emb is None else int(len(self.chunk_emb))
        s["papers_loaded"] = len(self.papers)
        s["embed_backend"] = models.embed_backend()
        s["rerank_backend"] = models.rerank_kind()
        prev = {}
        try:  # which model actually produced the vectors on disk
            prev = json.loads(VECTORS_SIG.read_text()) if VECTORS_SIG.exists() else {}
        except Exception:
            prev = {}
        cur = s["embed_backend"]
        if prev and (prev.get("backend") != cur["backend"] or prev.get("model") != cur["model"]):
            msg = (f"索引向量由 {prev.get('model')} 生成，当前 embedding 后端是 {cur['model']}；"
                   f"向量空间不同，请运行 ./run.sh index 重建（否则检索结果会失真）")
            s["warning"] = msg
            self.dim_warning = self.dim_warning or msg
        if self.dim_warning:
            s["warning"] = self.dim_warning
        return s

    # ---------------------------------------------------------------- helpers
    def allowed_mask(self, collection=None, year_from=None, year_to=None) -> np.ndarray | None:
        if not (collection or year_from or year_to):
            return None
        m = np.ones(len(self.papers), dtype=bool)
        if collection:
            cl = collection.lower()
            m &= np.array([any(cl == c.lower() or cl in c.lower() for c in p["collections"]) for p in self.papers])
        if year_from:
            m &= self.years >= int(year_from)
        if year_to:
            m &= (self.years <= int(year_to)) & (self.years > 0)
        return m

    def chunk_text(self, cid: int) -> str:
        r = self.db.execute("SELECT text FROM chunks WHERE id=?", (int(cid),)).fetchone()
        return r[0] if r else ""

    def _bm25_chunks(self, q, n, mask):
        fq = fts_query(q)
        if not fq:
            return []
        with self._lock:
            rows = self.db.execute(
                "SELECT rowid FROM chunks_fts WHERE chunks_fts MATCH ? ORDER BY bm25(chunks_fts) LIMIT ?",
                (fq, n * (4 if mask is not None else 1))).fetchall()
        ids = [r[0] for r in rows]
        if mask is not None:
            ids = [c for c in ids if mask[self.chunk_paper[c]]]
        return ids[:n]

    def _bm25_papers(self, q, n, mask):
        fq = fts_query(q)
        if not fq:
            return []
        with self._lock:
            rows = self.db.execute(
                "SELECT rowid FROM papers_fts WHERE papers_fts MATCH ? ORDER BY bm25(papers_fts, 3.0, 1.0, 1.0) LIMIT ?",
                (fq, n * (4 if mask is not None else 1))).fetchall()
        ids = [r[0] for r in rows]
        if mask is not None:
            ids = [p for p in ids if mask[p]]
        return ids[:n]

    def _dense_chunks(self, qv, n, mask):
        if self.chunk_emb is None:
            return []
        if qv.shape[0] != self.chunk_emb.shape[1]:
            self.dim_warning = (f"index vectors have dim {self.chunk_emb.shape[1]} but the current embedding "
                                f"backend produces {qv.shape[0]} — run ./run.sh index to rebuild")
            return []
        s = self.chunk_emb @ qv
        if mask is not None:
            s = np.where(mask[self.chunk_paper[self.chunk_emb_ids]], s, -1)
        k = min(n, len(s))
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return [int(self.chunk_emb_ids[i]) for i in top if s[i] > -1]

    def _dense_papers(self, qv, n, mask):
        if self.paper_emb is None:
            return []
        if qv.shape[0] != self.paper_emb.shape[1]:
            return []
        s = self.paper_emb @ qv
        if mask is not None:
            s = np.where(mask, s, -1)
        top = np.argsort(-s)[:n]
        return [int(i) for i in top if s[i] > -1]

    # ---------------------------------------------------------------- search
    def search(self, query: str, keywords: list[str] | None = None, collection: str | None = None,
               year_from: int | None = None, year_to: int | None = None, top_k: int = 8,
               use_rerank: bool = True, use_dense: bool = True, use_bm25: bool = True,
               locate: bool = True) -> dict:
        t0 = time.time()
        self.maybe_reload()
        mask = self.allowed_mask(collection, year_from, year_to)
        queries = [query] + [k for k in (keywords or []) if k and k != query][:4]
        qvs = models.embed_query(queries) if use_dense else [None] * len(queries)

        rrf: dict[int, float] = {}
        cand_chunks: dict[int, list[int]] = {}   # paper -> candidate chunk ids (ranked)

        def add_list(paper_ids, weight=1.0):
            seen = set()
            r = 0
            for p in paper_ids:
                if p in seen:
                    continue
                seen.add(p)
                rrf[p] = rrf.get(p, 0.0) + weight / (RRF_K + r + 1)
                r += 1

        for qi, (q, qv) in enumerate(zip(queries, qvs)):
            w = 1.0 if qi == 0 else 0.6
            lists = []
            if use_dense:
                dc = self._dense_chunks(qv, 200, mask)
                lists.append(("dc", dc))
                add_list([int(self.chunk_paper[c]) for c in dc], w)
                add_list(self._dense_papers(qv, 60, mask), w)
            if use_bm25:
                bc = self._bm25_chunks(q, 200, mask)
                lists.append(("bc", bc))
                add_list([int(self.chunk_paper[c]) for c in bc], w * BM25_WEIGHT)
                add_list(self._bm25_papers(q, 60, mask), w * BM25_WEIGHT * 0.8)
            for _, lst in lists:
                for c in lst:
                    cand_chunks.setdefault(int(self.chunk_paper[c]), [])
                    if c not in cand_chunks[int(self.chunk_paper[c])]:
                        cand_chunks[int(self.chunk_paper[c])].append(c)

        ranked = sorted(rrf, key=lambda p: -rrf[p])
        t_retr = time.time() - t0
        pool = ranked[: top_k + 8]

        # passages for rerank: best chunk for every pooled paper (+ abstract and 2nd chunk for the head)
        passages = []  # (paper, chunk_id or -1, text)
        for j, p in enumerate(pool):
            pp = self.papers[p]
            cc = cand_chunks.get(p, [])
            if pp["abstract"] and (j < top_k or not cc):
                passages.append((p, -1, pp["abstract"]))
            for c in cc[: (2 if j < top_k // 2 else 1)]:
                passages.append((p, c, self.chunk_text(c)))
            if not pp["abstract"] and not cand_chunks.get(p):
                passages.append((p, -1, pp["title"]))

        if use_rerank and passages:
            docs = [f"Title: {self.papers[p]['title']}\n{' '.join(t.split()[:200])}" for p, _, t in passages]
            scores = models.rerank(query, docs)
        else:
            scores = np.array([rrf[p] for p, _, _ in passages])
        t_rr = time.time() - t0 - t_retr

        by_paper: dict[int, list] = {}
        for (p, c, t), s in zip(passages, scores):
            by_paper.setdefault(p, []).append((float(s), c, t))
        final = sorted(by_paper, key=lambda p: (-max(x[0] for x in by_paper[p]), -rrf[p]))[:top_k]

        results = []
        kind = models.rerank_kind() if use_rerank else "none"
        # cross-encoder logits are unbounded (+/-10), so report them relative to the best hit of this
        # query; the LLM reranker already returns 0..1. Both are "relative match" for the UI only.
        top_score = max((max(x[0] for x in by_paper[p]) for p in final), default=1.0) or 1.0
        for rank, p in enumerate(final):
            pp = self.papers[p]
            ev = sorted(by_paper[p], key=lambda x: -x[0])
            best = ev[0][0]
            evid = []
            for s, c, t in ev[:2]:
                if c >= 0:
                    page = locate_page(pp["pdf_path"], t) if locate else None
                    evid.append({"text": t, "score": s, "chunk_id": c, "page": page, "kind": "passage"})
                else:
                    evid.append({"text": t, "score": s, "chunk_id": None, "page": None, "kind": "abstract"})
            results.append({
                **self.public(pp),
                "rank": rank + 1,
                "score": best,
                # `score` keeps the raw backend score; `relevance` is always relative to the best hit
                # of this query so the UI bar is comparable across backends (some rerank APIs return
                # unnormalised scores well below 1, e.g. nvidia nemotron gives 0.08-0.11).
                "relevance": (max(best / top_score, 0.0) if kind in ("local", "api") else
                              (float(best) if kind == "llm" else None)),
                "evidence": evid,
                "links": zotero_links(pp, next((e["page"] for e in evid if e["page"]), None)),
            })
        return {
            "query": query, "keywords": queries[1:], "filters": {"collection": collection, "year_from": year_from, "year_to": year_to},
            "results": results,
            "timing": {"retrieve_s": round(t_retr, 3), "rerank_s": round(t_rr, 3), "total_s": round(time.time() - t0, 3)},
            "coverage": {"chunks_embedded": 0 if self.chunk_emb is None else int(len(self.chunk_emb)),
                         "chunks_total": int(len(self.chunk_paper))},
        }

    def paper_passages(self, key: str, question: str, k: int = 4) -> dict | None:
        """Most relevant passages inside one paper for a question (for deep-dive questions)."""
        i = self.key2idx.get(key)
        if i is None:
            return None
        pp = self.papers[i]
        ids = [r[0] for r in self.db.execute("SELECT id FROM chunks WHERE paper=? ORDER BY ord", (i,))]
        if not ids:
            return {**self.public(pp), "passages": []}
        texts = [self.chunk_text(c) for c in ids]
        docs = [f"Title: {pp['title']}\n{' '.join(t.split()[:240])}" for t in texts]
        if len(docs) > 14:  # prefilter with dense / lexical overlap to keep rerank cheap
            qv = models.embed_query(question)
            if self.chunk_emb is not None:
                pos = {int(c): j for j, c in enumerate(self.chunk_emb_ids)}
                pre = [(float(self.chunk_emb[pos[c]] @ qv) if c in pos else -1.0) for c in ids]
            else:
                qs = set(_norm(question).split())
                pre = [len(qs & set(_norm(t).split())) for t in texts]
            keep = sorted(range(len(ids)), key=lambda j: -pre[j])[:14]
        else:
            keep = list(range(len(ids)))
        sc = models.rerank(question, [docs[j] for j in keep])
        order = [keep[j] for j in np.argsort(-sc)[:k]]
        out = []
        for j, s in zip(order, sorted(sc, reverse=True)[:k]):
            out.append({"text": texts[j], "score": float(s), "chunk_id": ids[j],
                        "page": locate_page(pp["pdf_path"], texts[j]), "kind": "passage"})
        return {**self.public(pp), "passages": out,
                "links": zotero_links(pp, next((e["page"] for e in out if e["page"]), None))}

    def search_metadata(self, title=None, author=None, year=None, collection=None, tag=None, limit=15,
                        sort="dateAdded") -> list[dict]:
        out = []
        for p in self.papers:
            if title and title.lower() not in p["title"].lower():
                continue
            if author and not any(author.lower() in a.lower() for a in p["authors"]):
                continue
            if year and str(year) != p["year"]:
                continue
            if collection and not any(collection.lower() in c.lower() for c in p["collections"]):
                continue
            if tag and not any(tag.lower() in t.lower() for t in p["tags"]):
                continue
            out.append(p)
        if sort == "year":
            out.sort(key=lambda p: p["year"] or "0", reverse=True)
        else:
            out.sort(key=lambda p: p["dateAdded"], reverse=True)
        return [{**self.public(p), "links": zotero_links(p)} for p in out[:limit]]

    def get(self, key: str) -> dict | None:
        i = self.key2idx.get(key)
        return None if i is None else self.papers[i]

    @staticmethod
    def public(p: dict) -> dict:
        return {k: p[k] for k in ("key", "title", "authors", "year", "venue", "type", "doi", "arxiv", "url",
                                  "abstract", "collections", "pdf_key", "citationKey")} | {
            "has_pdf": bool(p.get("pdf_path"))}


_LIB = None


def library() -> Library:
    global _LIB
    if _LIB is None:
        _LIB = Library()
    return _LIB
