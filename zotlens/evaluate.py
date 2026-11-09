"""Retrieval evaluation with an automatically generated test set (no manual labels, training-free).

1. Sample N papers; pick one body passage per paper; ask the LLM to write a question a researcher
   would ask that this passage answers WITHOUT copying title words (gold = that paper).
   Half of the questions are written in Chinese to test cross-lingual retrieval.
2. Compare retrieval variants (ablation): Zotero native full-text BM25, BM25 (ours, chunk-level),
   dense paper-level (title+abstract, ~ existing zotero-mcp semantic search), dense chunk-level,
   hybrid RRF, hybrid + rerank.
3. Report Recall@1/5/10, MRR@10 and latency.

Usage: python -m zotlens.evaluate --n 100 [--regen]
"""
from __future__ import annotations

import argparse
import json
import random
import sqlite3
import time

import numpy as np

from .config import DATA_DIR, ZOTERO_DIR, llm_config
from .retriever import fts_query, library

QSET = DATA_DIR / "eval_questions.json"


def gen_questions(n: int, seed: int = 0) -> list[dict]:
    from openai import OpenAI
    import httpx
    cfg = llm_config()
    client = OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"], http_client=httpx.Client(trust_env=False, timeout=60))
    lib = library()
    rng = random.Random(seed)
    covered = set(int(lib.chunk_paper[c]) for c in (lib.chunk_emb_ids if lib.chunk_emb is not None else []))
    cands = [i for i, p in enumerate(lib.papers) if p["text_cache"] and i in covered]
    rng.shuffle(cands)
    out = []
    for i in cands:
        if len(out) >= n:
            break
        ids = [r[0] for r in lib.db.execute("SELECT id FROM chunks WHERE paper=? AND ord BETWEEN 2 AND 20", (i,))]
        if not ids:
            continue
        cid = rng.choice(ids)
        passage = lib.chunk_text(cid)
        lang = "Chinese" if len(out) % 2 else "English"
        prompt = (f"Here is a passage from a research paper:\n\n{passage[:2500]}\n\n"
                  f"Write ONE specific question in {lang} that a researcher might ask and that this passage answers. "
                  "Do not mention the paper's title, method name, authors, or dataset names verbatim; paraphrase. "
                  "Output only the question.")
        try:
            r = client.chat.completions.create(model=cfg["model"], messages=[{"role": "user", "content": prompt}],
                                               max_tokens=120, temperature=0.7, extra_body=cfg["extra_body"] or None)
            q = r.choices[0].message.content.strip().strip('"')
        except Exception as e:  # noqa: BLE001
            print("gen failed", e)
            continue
        out.append({"q": q, "lang": lang, "gold": lib.papers[i]["key"], "chunk": cid, "title": lib.papers[i]["title"]})
        print(f"[{len(out)}/{n}] {q}", flush=True)
    return out


class ZoteroNativeBM25:
    """Baseline: Zotero 10's own full-text index (~/Zotero/fulltext.sqlite, FTS5), attachment -> parent paper."""

    def __init__(self):
        import shutil, tempfile
        from pathlib import Path
        src = ZOTERO_DIR / "fulltext.sqlite"
        self.ok = src.exists()
        if not self.ok:
            return
        dst = Path(tempfile.gettempdir()) / "zotlens_fulltext.sqlite"
        shutil.copy2(src, dst)
        self.db = sqlite3.connect(dst)
        lib = library()
        from .zotero_data import snapshot_db
        snap = sqlite3.connect(f"file:{snapshot_db()}?mode=ro", uri=True)
        self.att2paper = {}
        for att_id, parent in snap.execute("SELECT itemID, parentItemID FROM itemAttachments WHERE parentItemID IS NOT NULL"):
            key = snap.execute("SELECT key FROM items WHERE itemID=?", (parent,)).fetchone()
            if key and key[0] in lib.key2idx:
                self.att2paper[att_id] = lib.key2idx[key[0]]

    def search(self, q, k=10):
        fq = fts_query(q)
        if not self.ok or not fq:
            return []
        rows = self.db.execute("SELECT rowid FROM fulltextContent WHERE fulltextContent MATCH ? "
                               "ORDER BY bm25(fulltextContent) LIMIT 50", (fq,)).fetchall()
        out = []
        for (rid,) in rows:
            p = self.att2paper.get(rid)
            if p is not None and p not in out:
                out.append(p)
        return out[:k]


def variants(lib, zn):
    from . import models

    def papers_of(chunk_ids):
        out = []
        for c in chunk_ids:
            p = int(lib.chunk_paper[c])
            if p not in out:
                out.append(p)
        return out

    def keys(idx):
        return [lib.papers[i]["key"] for i in idx[:10]]

    return {
        "Zotero native full-text (BM25)": lambda q: keys(zn.search(q)),
        "BM25 chunks (ours)": lambda q: keys(papers_of(lib._bm25_chunks(q, 200, None))),
        "Dense paper-level (title+abstract)": lambda q: keys(lib._dense_papers(models.embed_query(q), 10, None)),
        "Dense chunk-level": lambda q: keys(papers_of(lib._dense_chunks(models.embed_query(q), 200, None))),
        "Hybrid RRF (no rerank)": lambda q: [r["key"] for r in lib.search(q, top_k=10, use_rerank=False, locate=False)["results"]],
        "Hybrid RRF + rerank (ZotLens)": lambda q: [r["key"] for r in lib.search(q, top_k=10, locate=False)["results"]],
    }


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--regen", action="store_true")
    args = ap.parse_args(argv)
    lib = library()
    if args.regen or not QSET.exists():
        qs = gen_questions(args.n)
        QSET.write_text(json.dumps(qs, ensure_ascii=False, indent=1))
    qs = json.loads(QSET.read_text())[: args.n]
    zn = ZoteroNativeBM25()
    rows = []
    for name, fn in variants(lib, zn).items():
        ranks, lat, by_lang = [], [], {"English": [], "Chinese": []}
        for item in qs:
            t = time.time()
            res = fn(item["q"])
            lat.append(time.time() - t)
            gold = {item["gold"]} | set(lib.papers[lib.key2idx[item["gold"]]].get("dup_keys", []))
            r = next((i + 1 for i, k in enumerate(res) if k in gold), None)
            ranks.append(r)
            by_lang[item["lang"]].append(r)

        def rec(rs, k):
            return float(np.mean([x is not None and x <= k for x in rs])) if rs else float("nan")
        row = {"method": name, "R@1": rec(ranks, 1), "R@5": rec(ranks, 5), "R@10": rec(ranks, 10),
               "MRR@10": float(np.mean([1 / x if x else 0 for x in ranks])),
               "R@10 (EN q)": rec(by_lang["English"], 10), "R@10 (ZH q)": rec(by_lang["Chinese"], 10),
               "latency_ms": 1000 * float(np.median(lat))}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    (DATA_DIR / "eval_results.json").write_text(json.dumps({"n": len(qs), "rows": rows}, indent=1, ensure_ascii=False))
    hdr = "| Method | R@1 | R@5 | R@10 | MRR@10 | R@10 EN-q | R@10 ZH-q | Median latency |"
    lines = [hdr, "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['method']} | {r['R@1']:.2f} | {r['R@5']:.2f} | {r['R@10']:.2f} | {r['MRR@10']:.3f} | "
                     f"{r['R@10 (EN q)']:.2f} | {r['R@10 (ZH q)']:.2f} | {r['latency_ms']:.0f} ms |")
    md = f"Retrieval evaluation, N={len(qs)} auto-generated questions (half Chinese), gold = source paper.\n\n" + "\n".join(lines)
    (DATA_DIR / "eval_results.md").write_text(md + "\n")
    print("\n" + md)


if __name__ == "__main__":
    main()
