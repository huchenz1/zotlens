"""Central configuration.

Order of precedence: UI settings (`data/settings.json`, edited in the app) > process environment >
`.env` file > profile default. `.env` and `settings.json` are both gitignored.

Profiles (`ZOTLENS_PROFILE`):
  cloud (default when an OpenRouter key is present)  OpenRouter embeddings + rerank, no local models
  local                                             Qwen3 embedding + reranker on MPS/CUDA, fully offline
"""
import json
import os
from pathlib import Path

from . import settings as _settings

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    """Minimal .env reader (KEY=VALUE, # comments) so we do not add a dependency."""
    if not path.exists():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, val = line.split("=", 1)
        os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv(ROOT / ".env")

DATA_DIR = Path(os.environ.get("ZOTLENS_DATA", ROOT / "data"))
STATIC_DIR = ROOT / "static"

ZOTERO_DIR = Path(os.environ.get("ZOTERO_DIR", Path.home() / "Zotero"))
ZOTERO_SQLITE = ZOTERO_DIR / "zotero.sqlite"
ZOTERO_STORAGE = ZOTERO_DIR / "storage"
ZOTERO_API = os.environ.get("ZOTERO_API", "http://localhost:23119/api")

MODEL_DIR = Path(os.environ.get("ZOTLENS_MODELS", Path.home() / "models" / "zotlens"))
EMBED_MODEL = MODEL_DIR / "Qwen3-Embedding-0.6B"
RERANK_MODEL = MODEL_DIR / "Qwen3-Reranker-0.6B"

# ------------------------------------------------------------------ profile selection
OPENROUTER_KEY = os.environ.get("OPENROUTER_API_KEY", "")
OPENROUTER_BASE = os.environ.get("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")

# Default is `cloud`: nothing to download, one API key is enough. `local` is opt-in
# (ZOTLENS_PROFILE=local) because it needs ~2.2 GB of models.
PROFILE = os.environ.get("ZOTLENS_PROFILE", "cloud").lower()

# ------------------------------------------------------------------ retrieval backends
# embedding: local | api        rerank: local | api | llm | none
if PROFILE == "cloud":
    _embed_default, _rerank_default = "api", "api"
    _embed_base, _embed_model = OPENROUTER_BASE, "qwen/qwen3-embedding-8b"
    _embed_dim = 1024                      # Qwen3-8B supports 64..4096; 1024 keeps the index small
    _rerank_base, _rerank_model = OPENROUTER_BASE, "cohere/rerank-5-fast"
else:
    _embed_default, _rerank_default = "local", "local"
    _embed_base, _embed_model, _embed_dim = "https://api.openai.com/v1", "text-embedding-3-small", 0
    _rerank_base, _rerank_model = "https://api.siliconflow.cn/v1", "BAAI/bge-reranker-v2-m3"

EMBED_BACKEND = os.environ.get("ZOTLENS_EMBED", _embed_default).lower()
EMBED_API_BASE = os.environ.get("ZOTLENS_EMBED_API_BASE", _embed_base)
EMBED_API_MODEL = os.environ.get("ZOTLENS_EMBED_API_MODEL", _embed_model)
EMBED_API_KEY = os.environ.get("ZOTLENS_EMBED_API_KEY") or OPENROUTER_KEY
EMBED_API_DIM = int(os.environ.get("ZOTLENS_EMBED_API_DIM", str(_embed_dim)))  # 0 = model default
EMBED_API_BATCH = int(os.environ.get("ZOTLENS_EMBED_API_BATCH", 64))
EMBED_API_WORKERS = int(os.environ.get("ZOTLENS_EMBED_API_WORKERS", 4))  # parallel requests when indexing
EMBED_QUERY_PREFIX = os.environ.get("ZOTLENS_EMBED_QUERY_PREFIX", "")

RERANK_BACKEND = os.environ.get("ZOTLENS_RERANK", _rerank_default).lower()
RERANK_API_BASE = os.environ.get("ZOTLENS_RERANK_API_BASE", _rerank_base)
RERANK_API_MODEL = os.environ.get("ZOTLENS_RERANK_API_MODEL", _rerank_model)
RERANK_API_KEY = os.environ.get("ZOTLENS_RERANK_API_KEY") or OPENROUTER_KEY

# ------------------------------------------------------------------ UI overrides
def apply_settings(values: dict | None = None) -> None:
    """Push settings edited in the UI onto this module. Called on import and after each save, so a
    change takes effect without restarting the server."""
    global PROFILE, EMBED_BACKEND, EMBED_API_BASE, EMBED_API_MODEL, EMBED_API_KEY, EMBED_API_DIM
    global RERANK_BACKEND, RERANK_API_BASE, RERANK_API_MODEL, RERANK_API_KEY

    s = _settings.load() if values is None else values
    if not s:
        return
    base = s.get("openrouter_base") or OPENROUTER_BASE
    key = s.get("openrouter_key") or OPENROUTER_KEY
    if base or key:
        PROFILE = "cloud"
        EMBED_BACKEND = "api"
        EMBED_API_BASE = base
        EMBED_API_KEY = key or EMBED_API_KEY
        if s.get("embed_model"):
            EMBED_API_MODEL = s["embed_model"]
        if s.get("embed_dim"):
            EMBED_API_DIM = int(s["embed_dim"])
        RERANK_BACKEND = "api"
        RERANK_API_BASE = base
        RERANK_API_KEY = key or RERANK_API_KEY
        if s.get("rerank_model"):
            RERANK_API_MODEL = s["rerank_model"]


apply_settings()


# ------------------------------------------------------------------ chunking
CHUNK_WORDS = int(os.environ.get("ZOTLENS_CHUNK_WORDS", 300))
CHUNK_OVERLAP = int(os.environ.get("ZOTLENS_CHUNK_OVERLAP", 50))
MAX_CHUNKS_PER_PAPER = int(os.environ.get("ZOTLENS_MAX_CHUNKS", 60))


def llm_config() -> dict:
    """OpenAI-compatible chat endpoint used by the agent.

    ZOTLENS_LLM=deepseek (default) | local
    The DeepSeek key comes from DEEPSEEK_API_KEY / .env; as a convenience it can also be
    read from the WorkBuddy model registry on this machine, but it is never stored in code.
    """
    s = _settings.load()
    backend = (s.get("chat_backend") or os.environ.get("ZOTLENS_LLM", "deepseek")).lower()
    if backend == "local":
        return {
            "backend": "local",
            "base_url": s.get("chat_base") or os.environ.get("ZOTLENS_LOCAL_URL", "http://127.0.0.1:8081/v1"),
            "api_key": "local-no-auth",
            "model": s.get("chat_model") or os.environ.get("ZOTLENS_LOCAL_MODEL", ""),
            "extra_body": {},
        }
    key = s.get("chat_key") or os.environ.get("DEEPSEEK_API_KEY", "")
    if not key:
        reg = Path.home() / ".workbuddy" / "models.json"
        if reg.exists():
            try:
                for m in json.loads(reg.read_text()):
                    if "deepseek" in m.get("id", ""):
                        key = m.get("apiKey", "")
                        break
            except Exception:
                pass
    return {
        "backend": backend,
        "base_url": s.get("chat_base") or os.environ.get("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        "api_key": key,
        "model": s.get("chat_model") or os.environ.get("DEEPSEEK_MODEL", "deepseek-v4-flash"),
        "extra_body": {"thinking": {"type": "disabled"}},
    }


def problems() -> list[str]:
    """Human-readable setup problems, so the UI can explain what is missing instead of failing deep
    inside a request. Empty list = ready to run."""
    out: list[str] = []
    if PROFILE == "cloud":
        if not EMBED_API_KEY:
            out.append("未配置 OPENROUTER_API_KEY：复制 .env.example 为 .env 并填入 key（云端模式不需要下载任何模型）")
        if not RERANK_API_KEY:
            out.append("未配置 rerank 的 key（云端模式下与 OPENROUTER_API_KEY 相同即可）")
    else:
        missing = [m.name for m in (EMBED_MODEL, RERANK_MODEL) if not Path(m).exists()]
        if missing:
            out.append(f"本地模型缺失：{', '.join(missing)}（或改用默认云端模式：删掉 ZOTLENS_PROFILE=local）")
    if not llm_config().get("api_key"):
        out.append("未配置 DEEPSEEK_API_KEY：聊天模型不可用（检索仍可用）")
    return out


def summary() -> dict:
    """Non-secret view of the active configuration (used by /api/status and preflight)."""
    return {
        "profile": PROFILE,
        "embed": {"backend": EMBED_BACKEND,
                  "model": EMBED_API_MODEL if EMBED_BACKEND == "api" else EMBED_MODEL.name,
                  "base": EMBED_API_BASE if EMBED_BACKEND == "api" else str(EMBED_MODEL),
                  "dim": EMBED_API_DIM,
                  "key_set": bool(EMBED_API_KEY)},
        "rerank": {"backend": RERANK_BACKEND,
                   "model": RERANK_API_MODEL if RERANK_BACKEND == "api" else RERANK_MODEL.name,
                   "key_set": bool(RERANK_API_KEY)},
    }
