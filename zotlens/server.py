"""FastAPI backend: chat (SSE), direct search, PDF preview, open-in-Zotero, BibTeX."""
from __future__ import annotations

import asyncio
import json
import subprocess
import threading

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .agent import SESSIONS, chat, get_session
from .config import STATIC_DIR, ZOTERO_API, ZOTERO_STORAGE, llm_config
from .retriever import library

app = FastAPI(title="ZotLens")


@app.on_event("startup")
def _warm():
    def w():
        from . import models
        lib = library()
        models.embed_query("warm up")
        models.rerank("warm up", ["warm up"])
        print(f"[zotlens] ready: {len(lib.papers)} papers", flush=True)
    threading.Thread(target=w, daemon=True).start()


@app.get("/", response_class=HTMLResponse)
def index():
    return (STATIC_DIR / "index.html").read_text()


@app.get("/api/status")
def status():
    lib = library()
    lib.maybe_reload()
    zotero_items, zotero_ver = None, None
    try:
        r = httpx.get(f"{ZOTERO_API}/users/0/items/top", params={"limit": 1}, timeout=2.5, trust_env=False)
        if r.status_code == 200:
            zotero_items = int(r.headers.get("Total-Results") or 0) or None
            zotero_ver = int(r.headers.get("Last-Modified-Version") or 0) or None
    except Exception:
        pass
    cfg = llm_config()
    idx = lib.status()
    stale = None
    if zotero_ver:
        stale = {"zotero_version": zotero_ver, "index_version": idx.get("zotero_version"),
                 "changed": idx.get("zotero_version") != zotero_ver,
                 "zotero_items": zotero_items, "indexed": len(lib.papers),
                 "index_updated": idx.get("updated")}
    return {"index": idx, "zotero_running": zotero_ver is not None, "stale": stale,
            "llm": {"backend": cfg["backend"], "model": cfg["model"] or "auto"},
            "collections": lib.collections}


class ChatReq(BaseModel):
    session_id: str | None = None
    message: str
    attached: list[int] | None = None


@app.post("/api/chat")
async def chat_ep(req: ChatReq):
    sess = get_session(req.session_id)
    q: asyncio.Queue = asyncio.Queue()
    loop = asyncio.get_running_loop()

    def worker():
        try:
            for ev in chat(sess, req.message, req.attached):
                loop.call_soon_threadsafe(q.put_nowait, ev)
        except Exception as e:  # noqa: BLE001
            loop.call_soon_threadsafe(q.put_nowait, {"type": "error", "text": str(e)})
        loop.call_soon_threadsafe(q.put_nowait, None)

    threading.Thread(target=worker, daemon=True).start()

    async def gen():
        yield f"data: {json.dumps({'type': 'session', 'session_id': sess.id})}\n\n"
        while True:
            ev = await q.get()
            if ev is None:
                break
            yield f"data: {json.dumps(ev, ensure_ascii=False, default=float)}\n\n"

    return StreamingResponse(gen(), media_type="text/event-stream",
                             headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"})


class SearchReq(BaseModel):
    query: str
    keywords: list[str] | None = None
    collection: str | None = None
    year_from: int | None = None
    year_to: int | None = None
    top_k: int = 8
    session_id: str | None = None


@app.post("/api/search")
def search_ep(req: SearchReq):
    """Retrieval only (no LLM). Results are registered in the session so citation numbers stay consistent."""
    r = library().search(req.query, req.keywords, req.collection, req.year_from, req.year_to, req.top_k)
    sess = get_session(req.session_id)
    sess.turn += 1
    r["cards"] = [sess.sources.add(x, x["evidence"], sess.turn, req.query) for x in r["results"]]
    r["session_id"] = sess.id
    sess.messages.append({"role": "user", "content": f"(searched the library for: {req.query})"})
    sess.messages.append({"role": "assistant", "content": "Results: " + "; ".join(
        f"[{c['n']}] {c['title']}" for c in r["cards"])})
    return r


@app.get("/api/session/{sid}/sources")
def sources(sid: str):
    s = SESSIONS.get(sid)
    return s.sources.all() if s else []


@app.get("/api/pdf/{att_key}")
def pdf(att_key: str):
    if not att_key.isalnum():
        raise HTTPException(400)
    d = (ZOTERO_STORAGE / att_key).resolve()
    if not str(d).startswith(str(ZOTERO_STORAGE.resolve())) or not d.is_dir():
        raise HTTPException(404)
    pdfs = sorted(d.glob("*.pdf"))
    if not pdfs:
        raise HTTPException(404)
    return FileResponse(pdfs[0], media_type="application/pdf",
                        headers={"Content-Disposition": "inline"})


class OpenReq(BaseModel):
    url: str


@app.post("/api/open")
def open_in_zotero(req: OpenReq):
    if not req.url.startswith("zotero://"):
        raise HTTPException(400)
    subprocess.Popen(["open", req.url])
    return {"ok": True}


@app.get("/api/bibtex/{key}")
def bibtex(key: str):
    if not key.isalnum():
        raise HTTPException(400)
    try:
        r = httpx.get(f"{ZOTERO_API}/users/0/items/{key}", params={"format": "bibtex"}, timeout=5, trust_env=False)
        if r.status_code == 200:
            return {"bibtex": r.text.strip()}
    except Exception:
        pass
    p = library().get(key)
    if not p:
        raise HTTPException(404)
    auth = " and ".join(p["authors"])
    return {"bibtex": f"@article{{{p['citationKey'] or key},\n  title={{{p['title']}}},\n  author={{{auth}}},\n"
                      f"  year={{{p['year']}}},\n  journal={{{p['venue']}}}\n}}"}


@app.post("/api/reset/{sid}")
def reset(sid: str):
    SESSIONS.pop(sid, None)
    return {"ok": True}


def main():
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=int(__import__("os").environ.get("ZOTLENS_PORT", 8765)))


if __name__ == "__main__":
    main()
