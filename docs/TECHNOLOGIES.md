# Technologies used

Every technique below is implemented in this repository; the "where" column points at the file that
does the work, and the "measured" column is what it cost or bought on a real 1,994-paper library
(68,090 passages). Numbers come from `docs/eval_results.md` and the timing logs in `data/`.

## 1. Retrieval

| Technique | How it is used | Where | Measured |
|---|---|---|---|
| **SQLite FTS5** (BM25) | Two virtual tables over the same content: `chunks_fts` (passage text) and `papers_fts` (title/authors/abstract, weighted 3:1:1). `tokenize='porter unicode61'`, so English morphology is handled (search → searches). | `build_index.py`, `retriever._bm25_*` | 8 ms per query; R@10 0.45 |
| **Sentence embeddings** | `Qwen3-Embedding-8B` over OpenRouter at `dimensions=1024` (cloud profile) or `Qwen3-Embedding-0.6B` on MPS/CUDA (local profile). Query side gets the model's instruction prefix. | `models.py` | see cost table |
| **Dense retrieval** | Vectors are L2-normalised `float16`, so cosine similarity is a single matrix–vector product. Two granularities: every passage (68k) and every paper (title + abstract). | `retriever._dense_chunks/_dense_papers` | 1.5 s/query (includes the API embedding round-trip) |
| **Reciprocal Rank Fusion** | Four ranked lists (BM25 chunks, BM25 papers, dense chunks, dense papers) are merged by `1/(k+rank)`, `k=60`. Fusion happens at paper level; a paper inherits its best passage. No score calibration needed between BM25 and cosine. | `retriever.search` | R@10 0.62 without rerank |
| **Cross-encoder reranking** | `cohere/rerank-5-fast` (cloud) / `Qwen3-Reranker-0.6B` (local). Reads query and passage jointly, scores ~20 candidates, cannot be pre-computed. | `models.rerank`, `retriever.search` | R@1 0.38 → **0.58**, MRR 0.46 → **0.61**; $0.00015 per query |
| **Incremental indexing** | Every passage is keyed by `sha1(title + text)`; a rebuild re-embeds only new hashes. A `vectors_model.json` signature records which model actually produced the vectors. | `build_index.py` | 6 new papers → 156 passages, 30 s |

Two ablations we ran and rejected (same 100 questions): down-weighting BM25 to 0.5 (R@10 0.65) and
dropping BM25 entirely (R@10 0.62) both lose to equal weights (0.68).

## 2. Data layer

| Technique | How it is used | Where |
|---|---|---|
| **Read-only Zotero access** | `zotero.sqlite` is copied to a snapshot before reading (Zotero may hold an exclusive lock); the live database is never opened for writing. | `zotero_data.py` |
| **Reusing Zotero's own text extraction** | Zotero already caches every PDF's text in `.zotero-ft-cache` next to the attachment. We read that instead of parsing PDFs, which removes an entire OCR/layout dependency from the index path. | `zotero_data.read_paper_text` |
| **Record linkage / dedupe** | Items are merged by DOI → arXiv id → normalised title. 2,548 raw items became 1,994 unique papers (560 duplicates removed). | `zotero_data.dedupe` |
| **Chunking** | 300-word windows with 50-word overlap, capped at 60 chunks per paper, plus a bibliography-stripping heuristic so reference lists do not pollute retrieval. | `zotero_data.chunk_text` |
| **PDF page localisation** | The passage is searched inside the PDF's pages with a normalised 7-gram probe (three probe offsets to survive header/footer noise), falling back to best 4-gram overlap. Output feeds `zotero://open-pdf/library/items/<key>?page=N`. | `retriever.locate_page` | 59/59 sampled passages resolved |

## 3. Agent

| Technique | How it is used | Where |
|---|---|---|
| **Tool calling (function calling)** | Five tools are given to an OpenAI-compatible chat model: `search_library`, `read_paper`, `find_papers`, `list_collections`, `load_skill`. The model decides whether the library is needed at all, and issues parallel calls when it needs several papers. | `agent.py`, `tools.py` |
| **Agent Skills** | `skills/*/SKILL.md` — YAML frontmatter (`name`, `description`) plus a step-by-step body. Only the descriptions sit in the system prompt; the body is loaded on demand through `load_skill`, which keeps the prompt small and the behaviour consistent (the same pattern as Anthropic's Agent Skills). | `skills/`, `tools.load_skills` |
| **Streaming** | Chat completions are streamed token by token, so the UI starts printing before the answer is complete. | `agent.chat` |
| **Citation registry** | Each paper gets a stable `[n]` for the whole session; evidence passages and page numbers travel with it, so the UI can render a source card per citation and the model can be told which numbers exist. | `tools.SourceRegistry` |
| **Guardrails** | Max 6 tool steps per turn, a 300 s wall-clock cap per model call, citation numbers validated against what tools actually returned. | `agent.py` |

## 4. Interfaces

| Technique | How it is used | Where |
|---|---|---|
| **SSE over HTTP** | The agent runs in a worker thread and pushes events (`status`, `tool`, `sources`, `token`, `done`) into an `asyncio.Queue`, which a FastAPI `StreamingResponse` forwards as `text/event-stream`. | `server.py` |
| **MCP (Model Context Protocol)** | The same four tools and the four skills are exposed over stdio and streamable HTTP, so Claude Desktop / Cursor / WorkBuddy can query the library with identical citations. Compatible with `mcp` 1.x (`FastMCP`) and 2.x (`MCPServer`). | `mcp_server.py` |
| **Single-page UI** | No build step: `static/index.html` with `marked` (Markdown), `DOMPurify` (sanitising model output), `KaTeX` (formulas), a hand-written SSE reader, and citation ↔ card linking. | `static/index.html` |
| **Frontend rendering** | Evidence snippets are windowed around the first query-term hit (not the start of the chunk), and relevance bars are normalised per query because rerank scores are not calibrated across models. | `static/index.html` |

## 5. Operations

| Technique | How it is used | Where |
|---|---|---|
| **Pluggable backends** | `ZOTLENS_EMBED=local|api`, `ZOTLENS_RERANK=local|api|llm|none`. Any OpenAI-compatible `/embeddings` and any `POST /rerank` provider works; the dimension is auto-probed and recorded. | `config.py`, `models.py` |
| **Profiles** | `cloud` (default, one API key, nothing to download) vs `local` (opt-in, ~2.2 GB of models, fully offline). | `config.py` |
| **Preflight** | `./run.sh check` verifies embedding, rerank and chat in one shot and reports missing keys before an index build. | `preflight.py` |
| **Evaluation harness** | Questions are generated from real passages (half English, half Chinese) without reusing titles, so keyword search cannot cheat; six retrieval designs are compared on Recall@1/5/10 and MRR. | `evaluate.py` |
| **Retry / backoff** | Provider rate limits (HTTP 429, per-model token quota) are retried 8 times with exponential backoff up to 60 s, honouring `Retry-After`; an interrupted index build resumes from the last checkpoint. | `models._embed_batch` |

## Cost of the whole library (measured)

| Item | Value |
|---|---|
| Passages embedded | 68,090 (≈ 36.7 M tokens) |
| Full index build | **$0.41**, ~25 min at 4 concurrent requests |
| Rerank per query (20 candidates, 7.6k tokens) | **$0.00015** |
| 1,000 papers, one-off | ≈ $0.21 |
