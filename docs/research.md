# 基于本地 Zotero 的问题驱动文献检索系统 — 实现调研

> 目标：用户用自然语言提问（如"masked diffusion 并行采样为什么会掉质量？"），系统自动从**本地 Zotero 库**中找出相关文献，给出排序列表 + 原文证据片段（含页码）+ 一键在 Zotero 中打开。
> 约束：COMP6708 A3 要求 training-free、可交互 UI、使用 RAG / MCP / Agent Skills 至少一种；11/25 现场 5 分钟 demo。

---

## 0. 结论先行

**推荐方案：Hybrid RAG（Zotero 自带 BM25 + 本地 dense embedding）+ reranker + 论文级聚合，封装为 MCP server，再用 Agent Skills 编排多步检索，前端 Streamlit/Gradio。**

关键判断（均已在本机实测或查官方文档确认）：

1. **数据层不用自己解析 PDF。** Zotero 已经把所有 PDF 抽成纯文本缓存（`storage/<KEY>/.zotero-ft-cache`，4189 份），直接读即可，省掉最慢、最易出错的一步。
2. **BM25 是现成的。** Zotero 10 的 `~/Zotero/fulltext.sqlite` 是 **SQLite FTS5 索引**（`fulltextContent`, tokenize=unicode61），可直接调 `bm25()`，实测一次查询 **15 ms**。
3. **但纯关键词不够。** 实测查询 `masked diffusion parallel sampling`，库里最贴题的 *Walk Fast but Be Careful: Understanding Parallel Sampling in Masked Diffusion* 不在 top-6，且结果出现重复条目 → 需要 dense 检索 + rerank + 去重，这正好是 demo 里可以讲的"为什么这样设计"。
4. **已有开源方案（zotero-mcp 等）能用但不够做作业。** 它们大多只做元数据级语义检索或把全文整篇塞给 LLM，缺少 chunk 级证据定位、hybrid 融合和评估——这是我们的创新点空间。

---

## 1. 你的 Zotero 库现状（实测，只读快照）

| 项 | 数值 |
|---|---|
| Zotero 版本 | 10.0.6 |
| 论文类条目 | preprint 1932 + journal 255 + conference 248 + 其他 ≈ **2.4k** |
| 有 abstract 的条目 | 2421 |
| PDF 附件 | 2563 |
| 全文已索引（FTS5 行数） | 4164 |
| `.zotero-ft-cache` 文本缓存 | 4189 份，均值 ~62 KB → 合计 ~260 MB 文本（估算 ~6000 万 tokens） |
| storage 总量 | 18 GB |
| Collections | 10 个（Generative Models 1069、LLM 624、RL 196、NN & Learning Theory 156 …） |
| Tags | 362（大多是 arXiv 分类自动标签） |
| 有 DOI 的条目 | 1968 |
| 用户批注 | 5 条（几乎没用，不作为主要信号） |
| 硬件 | Apple M5 Pro, 48 GB 内存 |

主题集中在 **生成模型 / LLM / diffusion LM / RL / post-training**，非常适合做 demo（问题好设计、评委能看懂）。

---

## 2. 数据接入方式对比

| 方式 | 说明 | 优点 | 缺点 | 用途 |
|---|---|---|---|---|
| **Zotero Local API** `http://localhost:23119/api/` | Zotero 7+ 内置，Web API v3 兼容；需在 设置→高级 勾选 "Allow other applications on this computer to communicate with Zotero"，否则 403 | 官方、读不需鉴权、不分页、有 `/items/<key>/fulltext`、`/items/<key>/file` 返回本地 `file://` 路径；Zotero 10+ 支持写（需授权） | **Zotero 必须在运行**（当前探测时未运行，返回连接失败） | 在线增量同步、获取最新元数据 |
| **直接读 `zotero.sqlite`** | 元数据（标题/作者/年份/DOI/collection/tag） | 批量快、离线 | Zotero 运行时可能锁库；schema 非公开 API，版本间可能变 → **只读复制快照**（`cp` 后用 `mode=ro`） | 首次全量建索引 |
| **`fulltext.sqlite`（FTS5）** | Zotero 10 的全文倒排索引 | 现成 BM25，15 ms | **contentless**（`content=''`）：只能拿到 rowid（=附件 itemID）和分数，拿不到原文片段 | 稀疏检索通道 |
| **`.zotero-ft-cache`** | 每个附件目录下的纯文本 | 免 PDF 解析 | 无页码信息；含页眉、参考文献等噪声 | chunk 化 + embedding 的原料 |
| 自己解析 PDF（PyMuPDF / GROBID / MinerU） | — | 能拿到页码、章节结构 | 2.5k PDF 慢 | **只对 top 结果按需做**，用于页码定位 |

**建议**：首次建索引走"SQLite 快照 + ft-cache"，在线增量走 Local API（按 `dateModified`/version 拉新条目），证据页码用 PyMuPDF 对命中论文做按需定位。

---

## 3. 现有开源方案调研

| 项目 | 形态 | 检索能力 | 局限 |
|---|---|---|---|
| **54yyyu/zotero-mcp**（Python，最流行） | MCP server | 关键词 + ChromaDB 语义检索；`update-db` 默认只索引元数据，`--fulltext` 才索引全文；embedding 可选默认/OpenAI/Gemini | 全文是整篇级向量，chunk 级证据定位弱；无 hybrid/rerank |
| **cookjohn/zotero-mcp** | Zotero 插件内置 MCP（Streamable HTTP, 端口 23120） | 语义检索（OpenAI/Ollama embedding + sqlite-vec），批注检索 | 依赖插件运行时；评估缺失 |
| **kujenga/zotero-mcp** | 轻量 Python MCP | search / metadata / fulltext 三个工具 | 只有 Zotero 自带 quicksearch |
| **tobyvee/mcp-zotero** | Go，只读 Local API | 关键词 | 同上 |
| **PaperQA2** `paperqa.contrib.ZoteroDB` | RAG 框架 | 质量高（含 rerank/contextual summary） | 走 **Web API 下载 PDF 重新解析**，官方自己说"a while since we've tested"；大库很慢、贵 |
| llm-for-zotero / Zotero GPT / Aria | Zotero 插件 | 对**单篇/选中**论文聊天 | 不做全库检索 |

Zotero 10 本身**没有 AI/语义检索**（官方 changelog 只增强了高级搜索、全文索引速度、CJK 短语匹配）。

**定位**：别 fork 现成项目当作业（创新分低）；可以参考 54yyyu 的工具命名，自己实现检索核心，做 baseline 对比时把"Zotero 自带关键词检索"和"zotero-mcp 元数据语义检索"当 baseline。

---

## 4. 推荐架构

```
┌──────────── UI (Streamlit / Gradio) ────────────┐
│ 提问框 · 过滤(collection/年份) · 结果卡片 · 证据高亮 · "在 Zotero 打开" │
└───────────────────────┬─────────────────────────┘
                        │
┌──────────── Agent 层 (LLM, training-free) ──────┐
│ Skill: lit-search   查询改写/拆解 → 调工具 → 判断是否够 → 再检索   │
│ Skill: compare      多篇对比表                                     │
│ Skill: related-work 生成带引用的 related work 草稿                 │
└───────────────────────┬─────────────────────────┘
                        │ MCP (FastMCP, stdio/HTTP)
┌──────────── MCP server: zotero-rag ──────────────┐
│ search_papers(query, k, filters)  → 论文列表 + 分数 + 命中片段     │
│ get_evidence(item_key, query)     → top 段落 + 页码                │
│ get_item(item_key)                → 元数据 / BibTeX                │
│ list_collections()                                                 │
│ open_in_zotero(item_key, page)    → zotero:// 链接                 │
└───────────────────────┬─────────────────────────┘
                        │
┌──────────── 检索核心 ────────────────────────────┐
│ ① BM25 : Zotero fulltext.sqlite FTS5（+ 自建 FTS5 on chunks）      │
│ ② Dense: Qwen3-Embedding-0.6B → FAISS / LanceDB（chunk + 摘要两级） │
│ ③ RRF 融合 → ④ Qwen3-Reranker-0.6B / bge-reranker-v2-m3 精排        │
│ ⑤ 论文级聚合（chunk→paper，max+top-3 mean）+ 去重（DOI/arXiv ID/标题）│
└───────────────────────┬─────────────────────────┘
                        │
   zotero.sqlite 快照 · .zotero-ft-cache · Local API(增量) · PDF(按需定位页码)
```

### 4.1 索引（离线，一次性 + 增量）

1. **元数据**：从 sqlite 快照取 `itemKey, title, creators, year, abstract, DOI, extra(arXiv ID), collections, tags, 附件 key`。
2. **去重**：按 DOI → arXiv ID → 归一化标题合并（实测 BM25 结果里已经出现同一论文两条）。
3. **文本清洗**：读 `.zotero-ft-cache`，截掉 `References`/`Bibliography` 之后的内容（否则检索会命中别人的参考文献列表），去页眉页脚。
4. **两级索引**：
   - **Paper-level**：`title + abstract` 一条向量（~2.4k 条，秒级）——快速召回、主题匹配；
   - **Chunk-level**：按段落切 ~400–512 tokens、overlap 64，每个 chunk 前缀拼上论文标题（contextual chunk）——证据定位。
5. **规模估算**（估算，未实测）：~260 MB 文本去掉参考文献与重复后约 8–12 万 chunk；1024 维 fp16 向量约 200–250 MB，FAISS Flat 内存可放下，查询 < 50 ms。
6. **Embedding 时间**：Qwen3-Embedding-0.6B 在 M5 Pro (MPS) 上预计数十分钟量级；放 A800 上几分钟。**建议在 A800 上建一次索引，产物拷回本机**，demo 时只做 query embedding（本地毫秒级）。
7. **增量同步**：记录上次索引的 `MAX(clientDateModified)`/item version，定时或启动时只处理新/改条目。

### 4.2 在线检索流程

1. **查询理解（LLM）**：把中文/口语问题改写成 2–4 个英文检索 query（多数论文是英文）+ 可选 HyDE（生成一段"假想摘要"做 dense 查询）；抽取过滤条件（"2024 年以后"、"在 LLM collection 里"）。
2. **召回**：每个 query 分别走 BM25（top-100）和 dense（paper-level top-50 + chunk-level top-200）。
3. **融合**：Reciprocal Rank Fusion（k=60），无需调权重，易解释。
4. **精排**：cross-encoder reranker 对 top-50 chunk 打分。
5. **聚合到论文**：paper score = max(chunk) + 0.3·mean(top-3 chunk)；每篇保留 1–2 个最佳证据片段。
6. **证据定位**：对最终 top-10 论文，用 PyMuPDF 在 PDF 中搜证据片段的前若干词，得到页码 → 生成 `zotero://open-pdf/library/items/<附件KEY>?page=N`，点击直接跳到 Zotero 阅读器对应页；`zotero://select/library/items/<KEY>` 在库中定位条目。
7. **生成（可选）**：LLM 基于证据片段回答，要求每句话带 `[n]` 引用，引用只允许来自检索结果（约束 hallucination）。

### 4.3 Agent Skills（体现"Agent"而非单轮 RAG）

- **lit-search**：检索 → LLM 判断"证据是否足以回答" → 不足则改写 query / 放宽过滤 / 沿已命中论文的关键词再检索（最多 3 轮）→ 输出。
- **compare-papers**：对选中的 3–5 篇，按"方法 / 数据 / 结论 / 局限"抽取并生成对比表。
- **related-work**：给一个研究主题，输出分主题段落的 related work 草稿，所有引用可点回 Zotero。

### 4.4 模型选择

| 组件 | 首选 | 备选 | 理由 |
|---|---|---|---|
| Embedding | Qwen3-Embedding-0.6B | bge-m3 | 中英跨语言（中文提问搜英文论文）、本地可跑；你们 RMDM 已在用，熟悉 |
| Reranker | Qwen3-Reranker-0.6B | bge-reranker-v2-m3 | 同系列、本地可跑 |
| LLM | DeepSeek / Qwen API | 本机 llama.cpp 跑 Qwen3 量化版 | API 稳；本地版作为"完全离线/隐私"卖点 |
| 向量库 | FAISS（最简） | LanceDB / sqlite-vec | sqlite-vec 能和 FTS5 放同一个文件，部署最简 |
| MCP | 官方 Python SDK FastMCP | — | 同一 server 既给自己的 UI 用，也能接 Claude Desktop / Cursor |

---

## 5. 评估方案（rubric 加分项，成本低）

**自动构造测试集**（无需人工标注，training-free）：
1. 从库里随机抽 100 篇论文，让 LLM 根据其**正文某段**（不是标题）生成一个"研究者会问的问题"，ground truth = 这篇论文；
2. 再人工写 20 个开放问题，人工标 2–5 篇相关论文（覆盖多篇相关的情况）。

**指标**：Recall@5/10、MRR、nDCG@10、平均延迟（P50/P95）。

**消融**（一张表就能撑起 demo 的"evaluation results"）：

| 方法 | R@10 | MRR | 延迟 |
|---|---|---|---|
| Zotero 自带关键词（BM25 over FTS5） | | | |
| Paper-level dense（≈ zotero-mcp 默认） | | | |
| Chunk-level dense | | | |
| Hybrid (RRF) | | | |
| Hybrid + rerank | | | |
| Hybrid + rerank + query 改写（agent） | | | |

预期：中文问题、同义改写类问题上，BM25 会明显掉，hybrid + rerank 最好——正好是 demo 的论点。

---

## 6. 风险与对策

| 风险 | 对策 |
|---|---|
| Zotero 运行时锁 `zotero.sqlite` | 只读复制快照后再读；在线部分走 Local API |
| Zotero schema 非公开 API | 读取逻辑集中在一个 adapter；Local API 作为兜底 |
| ft-cache 无页码 | 对 top 结果按需用 PyMuPDF 定位页码 |
| 参考文献段污染检索 | 截断 References 之后内容 |
| 重复条目 | DOI / arXiv ID / 标题三级去重 |
| 现场网络/API 限流 | 检索全本地；LLM 准备本地 llama.cpp 备份 + 录屏 |
| 隐私（PDF 发云端 LLM） | 只把 top 证据片段发给 LLM；提供全本地模式 |
| 演示机器不是你这台 | 索引产物 + 一个"示例 Zotero 快照"打包；或现场就用你的笔记本 |

---

## 7. 排期（到 11/25）

| 时间 | 交付 |
|---|---|
| 11/2–11/8 | 数据 adapter（sqlite 快照 + ft-cache + 去重 + 清洗）；BM25 通道；paper-level & chunk-level 索引在 A800 上建好 |
| 11/9–11/15 | Hybrid + rerank + 论文聚合 + 页码定位；MCP server；Streamlit 主界面跑通 |
| 11/16–11/21 | Agent skills（lit-search 多轮、compare、related-work）；评估集 + 消融表；增量同步 |
| 11/22–11/25 | UI 打磨、PPT（架构图 + 消融表 + 截图）、彩排、录备份视频 |

---

## 8. 5 分钟 demo 脚本建议

1. **0:00–0:40 问题**：几千篇 PDF，Zotero 只能关键词搜；展示在 Zotero 里搜一个中文问题 → 0 结果 / 结果不相关。
2. **0:40–2:30 主流程**：同一个问题输入系统 → 展示 agent 改写出的英文 query → 结果卡片（标题、年份、collection、分数、高亮证据段 + 页码）→ 点"在 Zotero 打开"直接跳到 PDF 对应页。
3. **2:30–3:30 Agent 能力**：选 3 篇点"对比" → 生成对比表；或"帮我写这部分 related work" → 带可点引用。
4. **3:30–4:20 MCP 复用**：同一个 MCP server 接到 Claude Desktop / Cursor 里直接问，说明可部署性。
5. **4:20–5:00 评估**：消融表 + 延迟数字。

---

## 9. 参考

- Zotero Local API 官方文档：https://www.zotero.org/support/dev/web_api/v3/local_api
- Zotero 10 changelog：https://www.zotero.org/support/changelog
- 54yyyu/zotero-mcp：https://github.com/54yyyu/zotero-mcp
- cookjohn/zotero-mcp：https://github.com/cookjohn/zotero-mcp
- kujenga/zotero-mcp：https://github.com/kujenga/zotero-mcp
- PaperQA2 Zotero 接入：https://docs.edisonscientific.com/paperqa/docs/tutorials/where_do_i_get_papers
- pyzotero（`local=True` 模式）：https://github.com/urschrei/pyzotero

*本机探测均在 `/tmp/zprobe/` 的只读副本上进行，未修改 `~/Zotero` 下任何文件。*
