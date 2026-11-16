"""FastAPI backend: chat (SSE), direct search, PDF preview, open-in-Zotero, BibTeX."""
from __future__ import annotations

import asyncio
import json
import time
import subprocess
import threading

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse, HTMLResponse, StreamingResponse
from pydantic import BaseModel

from .agent import SESSIONS, chat, get_session
from . import config, settings
from .config import STATIC_DIR, ZOTERO_API, ZOTERO_STORAGE, llm_config, problems, summary
from .retriever import library

app = FastAPI(title="ZotLens")


@app.on_event("startup")
def _warm():
    def w():
        from . import models
        lib = library()
        try:
            models.embed_query("warm up")
            models.rerank("warm up", ["warm up"])
        except Exception as exc:  # noqa: BLE001 - report, do not crash the server
            print(f"[zotlens] retrieval backend NOT ready: {exc}", flush=True)
        for msg in problems():
            print(f"[zotlens] setup: {msg}", flush=True)
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
            "problems": problems(), "profile": summary()["profile"],
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


# ------------------------------------------------------------------ settings (edited in the UI)
class SettingsReq(BaseModel):
    openrouter_base: str | None = None
    openrouter_key: str | None = None
    embed_model: str | None = None
    embed_dim: int | None = None
    rerank_model: str | None = None
    chat_backend: str | None = None
    chat_base: str | None = None
    chat_model: str | None = None
    chat_key: str | None = None


def _vector_signature() -> dict:
    from .build_index import VECTORS_SIG
    try:
        return json.loads(VECTORS_SIG.read_text())
    except Exception:
        return {}


@app.get("/api/settings")
def get_settings():
    sig = _vector_signature()
    return {"values": settings.public(),
            "defaults": config.DEFAULTS,
            "effective": {**summary(),
                          "embed_model": config.EMBED_API_MODEL,
                          "embed_dim": config.EMBED_API_DIM,
                          "rerank_model": config.RERANK_API_MODEL,
                          "base": config.EMBED_API_BASE,
                          "chat": {k: llm_config()[k] for k in ("backend", "base_url", "model")}},
            "index_vectors": {"model": sig.get("model"), "dim": sig.get("dim")},
            "can_edit_file": str(settings.PATH)}


@app.post("/api/settings")
def save_settings(req: SettingsReq):
    """Persist UI settings, apply them to the running process and smoke-test the backends."""
    from . import models
    before = (config.EMBED_API_MODEL, config.EMBED_API_DIM, config.EMBED_API_KEY)
    values = {k: v for k, v in req.model_dump().items() if v is not None}
    settings.save(values)
    config.apply_settings()
    models.reset()

    test = {"embedding": None, "rerank": None}
    try:
        v = models.embed_docs(["connectivity check"])
        test["embedding"] = {"ok": True, "dim": int(v.shape[1])}
    except Exception as exc:  # noqa: BLE001
        test["embedding"] = {"ok": False, "error": str(exc)[:300]}
    try:
        sc = models.rerank("connectivity check", ["connectivity check", "unrelated text"])
        test["rerank"] = {"ok": True, "scores": [round(float(x), 4) for x in sc]}
    except Exception as exc:  # noqa: BLE001
        test["rerank"] = {"ok": False, "error": str(exc)[:300]}

    sig = _vector_signature()
    rebuild = bool(sig) and (sig.get("model") != config.EMBED_API_MODEL or sig.get("dim") != config.EMBED_API_DIM)
    after = (config.EMBED_API_MODEL, config.EMBED_API_DIM, config.EMBED_API_KEY)
    return {"ok": True, "test": test, "rebuild_required": rebuild, "changed": before != after,
            "effective": summary()}


# ------------------------------------------------------------------ index maintenance
JOB = {"running": False, "log": [], "error": None, "finished": None}


@app.post("/api/index")
def start_index():
    """Rebuild/refresh the index in a background thread (incremental: only new passages are embedded)."""
    if JOB["running"]:
        return {"ok": True, "running": True, "already": True}
    JOB.update({"running": True, "log": [], "error": None, "finished": None})

    def work():
        import io
        import contextlib
        from . import build_index, models, retriever

        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf):
                build_index.main(["--batch", "16"])
            JOB["error"] = None
        except Exception as exc:  # noqa: BLE001
            JOB["error"] = f"{type(exc).__name__}: {exc}"
        finally:
            JOB["log"] = buf.getvalue().strip().splitlines()[-6:]
            JOB["running"] = False
            JOB["finished"] = time.strftime("%H:%M:%S")
            # the running server must adopt the new index: papers/order and vectors both changed
            try:
                models.reset()
                retriever._pdf_pages.cache_clear()
                library().load()
            except Exception as exc:  # noqa: BLE001
                JOB["error"] = (JOB["error"] or "") + f" reload failed: {exc}"

    threading.Thread(target=work, daemon=True).start()
    return {"ok": True, "running": True}


@app.get("/api/index/status")
def index_status():
    lib = library()
    st = lib.status()
    return {"running": JOB["running"], "error": JOB["error"], "finished": JOB["finished"],
            "log": JOB["log"], "stage": st.get("stage"), "chunks_embedded": st.get("chunks_embedded"),
            "chunks": st.get("chunks"), "papers": st.get("papers"),
            "stale": status().get("stale")}


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
