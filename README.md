# ZotLens — chat with your Zotero library

Ask a question in plain language; ZotLens searches the full text of your local Zotero library, answers with
citations, and shows **which paper** each claim comes from — down to the PDF page.

Built as a training-free LLM application (COMP6708 Assignment 3). It combines three techniques:

- **RAG** — hybrid retrieval (BM25 + dense) → cross-encoder reranking → grounded answer
- **MCP** — the same retrieval tools are exposed to any MCP client (Claude Desktop, Cursor, WorkBuddy)
- **Agent Skills** — `SKILL.md` procedures the agent loads on demand (search, compare, related work, deep read)

![chat](docs/screenshot-chat.png)

## Features

| | |
|---|---|
| **Chat about anything** | The agent decides whether the library is needed. Small talk gets a direct answer; research questions trigger retrieval first. |
| **Source panel** | Every retrieved paper becomes a card: `[n]` citation number, title/authors/year/venue, Zotero collections, relevance, and the evidence passage with its **PDF page**. Clicking `[n]` in the answer jumps to the card. |
| **Jump to the source** | Open the PDF in Zotero at the evidence page (`zotero://open-pdf/...?page=N`), preview it in-app, or copy BibTeX. |
| **Skills** | Multi-round search, comparison tables, related-work drafting, single-paper deep read. Select cards → 对比 / 写综述 / 基于所选提问. |
| **Search-only mode** | Retrieval + reranking without an LLM: fast, offline, and a safe fallback during a demo. |
| **MCP server** | Same tools, usable outside this UI. |

![compare](docs/screenshot-compare.png)

## Quick start

```bash
cp .env.example .env        # then fill in OPENROUTER_API_KEY and DEEPSEEK_API_KEY
python3 -m venv .venv && .venv/bin/pip install -r requirements.txt   # skip this if you have an env ready

./run.sh check              # verify embedding / rerank / chat backends before indexing
./run.sh index              # build the index (~20 min for 2k papers in the cloud profile)
./run.sh                    # http://127.0.0.1:8765
```

The app is usable as soon as BM25 and paper-level vectors exist; chunk vectors are hot-reloaded while being
built. Re-running `./run.sh index` is incremental — unchanged chunks reuse their vectors.

Requirements:

- **Zotero** running, with *Settings → Advanced → “Allow other applications on this computer to communicate
  with Zotero”* enabled (used for BibTeX and the status indicator; retrieval works without it).
- An **OpenRouter key** (`cloud` profile) or the two local models (`local` profile, see below).

## Configuration

All settings come from `.env` (gitignored) or the environment, in that order of precedence. Two profiles:

| Profile | Embedding | Rerank | Needs |
|---|---|---|---|
| `cloud` (default when `OPENROUTER_API_KEY` is set) | `qwen/qwen3-embedding-8b` (1024-d) | `cohere/rerank-5-fast` | one API key |
| `local` | Qwen3-Embedding-0.6B on MPS/CUDA | Qwen3-Reranker-0.6B | ~2.2 GB of models, fully offline |

```bash
# cloud (default)
OPENROUTER_API_KEY=sk-or-...

# local instead
ZOTLENS_PROFILE=local
ZOTLENS_MODELS=~/models/zotlens      # contains Qwen3-Embedding-0.6B/ and Qwen3-Reranker-0.6B/
```

Individual backends can be overridden independently, which is what makes other providers pluggable:

| Variable | Values | Default (`cloud` / `local`) |
|---|---|---|
| `ZOTLENS_EMBED` | `local` \| `api` | `api` / `local` |
| `ZOTLENS_EMBED_API_BASE` | any OpenAI-compatible base URL | `https://openrouter.ai/api/v1` |
| `ZOTLENS_EMBED_API_MODEL` | model id | `qwen/qwen3-embedding-8b` |
| `ZOTLENS_EMBED_API_DIM` | force output dimension (`0` = provider default) | `1024` / `0` |
| `ZOTLENS_EMBED_API_WORKERS` | parallel requests while indexing | `4` |
| `ZOTLENS_RERANK` | `local` \| `api` \| `llm` \| `none` | `api` / `local` |
| `ZOTLENS_RERANK_API_MODEL` | model id | `cohere/rerank-5-fast` |
| `ZOTLENS_LLM` | `deepseek` \| `local` | `deepseek` |

Any OpenAI-compatible `/embeddings` endpoint works (OpenAI, Jina, Together, DashScope, self-hosted vLLM);
rerank expects the `POST /rerank` shape used by Cohere, Jina, SiliconFlow and OpenRouter.

Two notes from testing:

- **DeepSeek has no embeddings endpoint.** `POST /embeddings` returns 404 and `/models` lists only chat models,
  so vectors need a separate provider.
- **Switching the embedding model invalidates the index.** A different model means a different vector space even
  when the dimension matches (Qwen3-0.6B and bge-m3 are both 1024-d), so the builder deletes stale vectors and
  re-embeds, and the UI warns when the on-disk vectors do not match the configured model.

## Architecture

```
Browser UI (static/index.html)  ── SSE ──  FastAPI (zotlens/server.py)
                                              │
                                   Agent loop (agent.py): OpenAI-compatible tool calling
                                              │
              tools.py: search_library · read_paper · find_papers · list_collections · load_skill
                    │                                   └── skills/*/SKILL.md  (Agent Skills)
                    ├── mcp_server.py  (same tools over MCP)
                    │
      retriever.py  hybrid retrieval
        ├ BM25 over 68k full-text chunks (SQLite FTS5, porter) + BM25 over title/abstract
        ├ Dense: chunk-level + paper-level (cosine, NumPy)
        ├ Reciprocal Rank Fusion (paper level) → cross-encoder rerank
        └ evidence → PDF page localisation (PyMuPDF) → zotero:// deep links
                    │
      build_index.py / zotero_data.py
        read-only snapshot of zotero.sqlite · Zotero full-text cache (.zotero-ft-cache)
        dedupe (DOI / arXiv / title) · bibliography stripping · 300-word chunks
        incremental: unchanged chunks reuse vectors (sha1)
```

Design choices worth knowing:

- **No PDF parsing at index time.** Zotero already extracted the text (`.zotero-ft-cache`); we reuse it and only
  locate page numbers on demand, for the passages actually shown.
- **Read-only on your library.** `zotero.sqlite` is copied to a snapshot before reading, because Zotero may hold
  a lock on it.
- **Duplicates merged** by DOI → arXiv id → normalised title (560 duplicates across 2,548 raw items).
- **Two-stage retrieval on purpose**: bi-encoders are cheap enough to score the whole library, cross-encoders are
  accurate but must be run per candidate — so recall 200, rerank ~20, show 5.

![pdf](docs/screenshot-pdf.png)

## Cost and latency (measured on 1,994 papers / 68,090 chunks)

| | Cloud (`qwen3-embedding-8b` + `cohere/rerank-5-fast`) | Local (Qwen3-0.6B pair) |
|---|---|---|
| Full index build | **$0.41** (≈ 36.7 M tokens), ~25 min at 4 workers | $0, ~60 min on M5 Pro (MPS) |
| Incremental update | cents | seconds |
| Rerank per query (20 candidates, 7.6k tokens) | **$0.00015** | $0 |
| Retrieval latency | 2.5–6 s | ~1.5 s |
| Offline | no | yes |

Indexing is latency-bound, so `ZOTLENS_EMBED_API_WORKERS` matters: 1 → 4 → 8 workers measured 40 / 60 / 85
chunks/s. Eight workers also hits the provider's per-model token quota (HTTP 429), which is why the default is 4
and the retry path backs off up to 60 s. A build interrupted by a rate limit resumes from the last checkpoint
without re-embedding finished chunks.

Reranker choice matters more than it looks: `cohere/rerank-5-fast` bills per token ($0.02/M → $0.00015/query),
while `cohere/rerank-4-fast` bills per search ($0.002) — 13× more for the same 20 candidates.

## Evaluation

`zotlens/evaluate.py` generates the test set automatically (no manual labels):

1. Sample N papers, take one body passage each, ask the LLM to write a question that the passage answers, half in
   Chinese. The gold answer is that paper.
2. Compare six variants: Zotero's native full-text BM25, our chunk BM25, dense paper-level, dense chunk-level,
   hybrid RRF, hybrid + rerank.

Reports Recall@1/5/10, MRR@10 and latency to `data/eval_results.md`.

```bash
./run.sh eval 100 --regen
```

## MCP

```json
{"mcpServers": {"zotlens": {"command": "/path/to/zotlens/run.sh", "args": ["mcp"]}}}
```

`./run.sh mcp-http` serves the same tools over streamable HTTP on port 8766.

## Layout

```
zotlens/            config · zotero_data · build_index · retriever · models · agent · tools
                    server (FastAPI) · mcp_server · preflight · evaluate
static/index.html   single-page UI (no build step)
skills/*/SKILL.md   Agent Skills, loaded via the load_skill tool
docs/               screenshots + the design research write-up
run.sh              serve · check · index · eval · mcp · status
```

## Limitations

- Retrieval is brute-force NumPy cosine: fine up to a few hundred thousand chunks, needs ANN (FAISS/HNSW) beyond.
- Sessions live in memory; restarting the server clears the conversation.
- Zotero annotations are ignored (only 5 in this library) — a strong relevance signal for heavier annotators.
- The evidence page is located by searching the passage text in the PDF; heavily reflowed or scanned PDFs may miss.
