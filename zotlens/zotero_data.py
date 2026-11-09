"""Read-only adapter for a local Zotero library.

We never open the live database for writing: zotero.sqlite is copied to a
snapshot first (Zotero may hold a lock on it), then queried. PDF text comes
from Zotero's own full-text cache (storage/<KEY>/.zotero-ft-cache), so no PDF
parsing is needed at index time.
"""
from __future__ import annotations

import re
import shutil
import sqlite3
import tempfile
from collections import defaultdict
from pathlib import Path

from .config import ZOTERO_SQLITE, ZOTERO_STORAGE

PAPER_TYPES_EXCLUDED = {"attachment", "note", "annotation"}
VENUE_FIELDS = ["publicationTitle", "proceedingsTitle", "conferenceName", "bookTitle",
                "websiteTitle", "repository", "publisher", "university"]
ARXIV_RE = re.compile(r"(?:arxiv[:/ .]*|abs/|pdf/|10\.48550/arxiv\.)(\d{4}\.\d{4,5})", re.I)


def snapshot_db(src: Path = ZOTERO_SQLITE) -> Path:
    dst = Path(tempfile.gettempdir()) / "zotlens_snapshot.sqlite"
    shutil.copy2(src, dst)
    wal = Path(str(src) + "-wal")
    if wal.exists() and wal.stat().st_size > 0:
        shutil.copy2(wal, Path(str(dst) + "-wal"))
    return dst


def norm_title(t: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", (t or "").lower())


def _arxiv_id(*vals: str) -> str | None:
    for v in vals:
        if v:
            m = ARXIV_RE.search(v)
            if m:
                return m.group(1)
    return None


def load_library(db_path: Path | None = None) -> dict:
    """Return {'papers': [...], 'collections': [...]} with duplicates merged."""
    db_path = db_path or snapshot_db()
    con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    q = lambda s, *a: con.execute(s, a).fetchall()

    deleted = {r[0] for r in q("SELECT itemID FROM deletedItems")}
    items = {
        r["itemID"]: dict(itemID=r["itemID"], key=r["key"], type=r["typeName"], dateAdded=r["dateAdded"])
        for r in q("""SELECT i.itemID, i.key, i.dateAdded, t.typeName FROM items i
                      JOIN itemTypes t USING(itemTypeID)""")
        if r["itemID"] not in deleted
    }

    fields = defaultdict(dict)
    for r in q("""SELECT d.itemID, f.fieldName, v.value FROM itemData d
                  JOIN fields f USING(fieldID) JOIN itemDataValues v USING(valueID)"""):
        fields[r["itemID"]][r["fieldName"]] = r["value"]

    creators = defaultdict(list)
    for r in q("""SELECT ic.itemID, c.firstName, c.lastName, c.fieldMode FROM itemCreators ic
                  JOIN creators c USING(creatorID) ORDER BY ic.itemID, ic.orderIndex"""):
        name = r["lastName"] if r["fieldMode"] == 1 else f"{r['firstName']} {r['lastName']}".strip()
        creators[r["itemID"]].append(name)

    coll_rows = q("SELECT collectionID, collectionName, parentCollectionID, key FROM collections")
    coll_name = {r["collectionID"]: r["collectionName"] for r in coll_rows}
    item_colls = defaultdict(list)
    for r in q("SELECT collectionID, itemID FROM collectionItems"):
        item_colls[r["itemID"]].append(coll_name.get(r["collectionID"], "?"))

    tags = defaultdict(list)
    for r in q("SELECT it.itemID, t.name FROM itemTags it JOIN tags t USING(tagID)"):
        tags[r["itemID"]].append(r["name"])

    attachments = defaultdict(list)
    for r in q("""SELECT a.itemID, a.parentItemID, a.contentType, a.path, a.linkMode, i.key
                  FROM itemAttachments a JOIN items i ON i.itemID = a.itemID"""):
        if r["parentItemID"] is None or r["itemID"] in deleted:
            continue
        cache = ZOTERO_STORAGE / r["key"] / ".zotero-ft-cache"
        path = None
        if r["path"] and r["path"].startswith("storage:"):
            path = ZOTERO_STORAGE / r["key"] / r["path"][len("storage:"):]
        attachments[r["parentItemID"]].append(dict(
            key=r["key"], itemID=r["itemID"], contentType=r["contentType"] or "",
            path=str(path) if path else None,
            cache=str(cache) if cache.exists() else None,
            cache_size=cache.stat().st_size if cache.exists() else 0,
        ))
    con.close()

    papers = []
    for iid, it in items.items():
        if it["type"] in PAPER_TYPES_EXCLUDED:
            continue
        f = fields.get(iid, {})
        title = f.get("title", "").strip()
        if not title:
            continue
        atts = attachments.get(iid, [])
        pdfs = [a for a in atts if a["contentType"] == "application/pdf"]
        # prefer the PDF with the largest extracted text, else any attachment with text
        best = max(pdfs, key=lambda a: a["cache_size"], default=None)
        if not best or best["cache_size"] < 500:
            best = max(atts, key=lambda a: a["cache_size"], default=best)
        venue = next((f[v] for v in VENUE_FIELDS if f.get(v)), "")
        papers.append(dict(
            key=it["key"], itemID=iid, type=it["type"], title=title,
            authors=creators.get(iid, []),
            year=(f.get("date", "")[:4] if f.get("date", "")[:4].isdigit() else ""),
            venue=venue, doi=f.get("DOI", ""), url=f.get("url", ""),
            arxiv=_arxiv_id(f.get("url", ""), f.get("extra", ""), f.get("DOI", ""), f.get("archiveID", "")),
            abstract=f.get("abstractNote", ""), citationKey=f.get("citationKey", ""),
            collections=sorted(set(item_colls.get(iid, []))), tags=tags.get(iid, []),
            dateAdded=it["dateAdded"],
            pdf_key=best["key"] if best else None,
            pdf_path=best["path"] if best and best["contentType"] == "application/pdf" else None,
            text_cache=best["cache"] if best else None,
            dup_keys=[],
        ))
    papers = dedupe(papers)
    collections = sorted(
        [{"name": n, "count": sum(n in p["collections"] for p in papers)} for n in set(coll_name.values())],
        key=lambda c: -c["count"])
    return {"papers": papers, "collections": collections}


def dedupe(papers: list[dict]) -> list[dict]:
    """Merge items sharing a DOI, arXiv id or normalised title (keep richest copy)."""
    parent = list(range(len(papers)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    seen = {}
    for i, p in enumerate(papers):
        for k in filter(None, [p["doi"].lower() and "doi:" + p["doi"].lower(),
                               p["arxiv"] and "arxiv:" + p["arxiv"],
                               len(norm_title(p["title"])) > 15 and "t:" + norm_title(p["title"])]):
            if k in seen:
                parent[find(i)] = find(seen[k])
            else:
                seen[k] = i
    groups = defaultdict(list)
    for i in range(len(papers)):
        groups[find(i)].append(papers[i])

    def richness(p):
        size = Path(p["text_cache"]).stat().st_size if p["text_cache"] else 0
        return (p["pdf_path"] is not None, size, len(p["abstract"]))

    out = []
    for g in groups.values():
        g.sort(key=richness, reverse=True)
        keep = dict(g[0])
        for other in g[1:]:
            keep["dup_keys"].append(other["key"])
            keep["collections"] = sorted(set(keep["collections"]) | set(other["collections"]))
            for fld in ("abstract", "venue", "doi", "arxiv", "year"):
                if not keep[fld] and other[fld]:
                    keep[fld] = other[fld]
        out.append(keep)
    return out


REF_HEAD = re.compile(r"\n\s*(?:\d+\.?\s*)?(references|bibliography|reference)\s*\n", re.I)
APPX_HEAD = re.compile(r"\n\s*(?:appendix|appendices|supplementary material|"
                       r"[A-H]\.?\s+(?:proof|additional|experiment|implementation|detail|derivation|"
                       r"hyperparameter|related|more|further|extended|theoretical|ablation))", re.I)


def clean_text(text: str) -> str:
    """Drop the bibliography (keeps appendix), normalise whitespace and hyphenation."""
    n = len(text)
    m = None
    for cand in REF_HEAD.finditer(text):
        if cand.start() > 0.3 * n:
            m = cand
            break
    if m:
        tail = text[m.end():]
        a = APPX_HEAD.search(tail)
        text = text[:m.start()] + ("\n" + tail[a.start():] if a else "")
    text = re.sub(r"-\n(?=[a-z])", "", text)          # de-hyphenate line breaks
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def read_paper_text(p: dict) -> str:
    if not p.get("text_cache"):
        return ""
    try:
        return clean_text(Path(p["text_cache"]).read_text(encoding="utf-8", errors="ignore"))
    except OSError:
        return ""


def chunk_text(text: str, words: int, overlap: int, max_chunks: int) -> list[str]:
    toks = text.split()
    if not toks:
        return []
    step = max(1, words - overlap)
    chunks = []
    for start in range(0, len(toks), step):
        piece = toks[start:start + words]
        if len(piece) < 40 and chunks:
            break
        chunks.append(" ".join(piece))
        if len(chunks) >= max_chunks or start + words >= len(toks):
            break
    return chunks
