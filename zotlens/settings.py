"""Runtime settings edited from the UI.

Stored as JSON next to the index (`data/settings.json`, gitignored). Values here take precedence
over environment variables, so a user can clone the repo, open the page and configure everything
without touching `.env` or exporting anything.

Deliberately dependency-free and free of imports from `config` (config imports this), so applying
settings is just assigning attributes on the config module — no process restart needed.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PATH = Path(os.environ.get("ZOTLENS_SETTINGS", ROOT / "data" / "settings.json"))

FIELDS = (
    "openrouter_base", "openrouter_key",
    "embed_model", "embed_dim", "rerank_model",
    "chat_backend", "chat_base", "chat_model", "chat_key",
)

_cached: dict | None = None


def load() -> dict:
    global _cached
    if _cached is None:
        try:
            _cached = json.loads(PATH.read_text()) if PATH.exists() else {}
        except Exception:
            _cached = {}
    return _cached


def save(values: dict) -> dict:
    """Persist only known fields; empty strings mean 'fall back to env / profile default'."""
    global _cached
    cur = dict(load())
    for k, v in values.items():
        if k not in FIELDS:
            continue
        if v is None:
            continue
        v = v.strip() if isinstance(v, str) else v
        if v == "":
            cur.pop(k, None)
        else:
            cur[k] = v
    PATH.parent.mkdir(parents=True, exist_ok=True)
    PATH.write_text(json.dumps(cur, indent=1, ensure_ascii=False))
    try:
        PATH.chmod(0o600)  # contains an API key
    except Exception:
        pass
    _cached = cur
    return cur


def get(key: str, default=None):
    return load().get(key, default)


def public() -> dict:
    """What the UI may see: the key itself is never returned, only whether one is set."""
    d = dict(load())
    for k in ("openrouter_key", "chat_key"):
        if k in d:
            d[k] = ""
            d[f"{k}_set"] = True
    return d
