# Retrieval evaluation (measured 2026-11-10)

100 automatically generated questions (half Chinese), gold = the paper the passage came from.

| Method | R@1 | R@5 | R@10 | MRR@10 | R@10 EN-q | R@10 ZH-q | Median latency |
|---|---|---|---|---|---|---|---|
| Zotero native full-text (BM25) | 0.19 | 0.31 | 0.36 | 0.244 | 0.70 | 0.02 | 1 ms |
| BM25 chunks (ours) | 0.29 | 0.43 | 0.45 | 0.343 | 0.90 | 0.00 | 8 ms |
| Dense paper-level (title+abstract) | 0.26 | 0.40 | 0.44 | 0.326 | 0.52 | 0.36 | 1243 ms |
| Dense chunk-level | 0.52 | 0.67 | 0.76 | 0.593 | 0.82 | 0.70 | 1512 ms |
| Hybrid RRF (no rerank) | 0.38 | 0.56 | 0.62 | 0.458 | 0.74 | 0.50 | 1273 ms |
| Hybrid RRF + rerank (ZotLens) | 0.58 | 0.63 | 0.68 | 0.609 | 0.80 | 0.56 | 2770 ms |

## Fusion ablations (same 100 questions)

| Variant | R@1 | R@5 | R@10 | MRR@10 |
|---|---|---|---|---|
| Hybrid RRF, BM25 weight 1.0 (shipped) | 0.58 | 0.63 | 0.68 | 0.609 |
| Hybrid RRF, BM25 weight 0.5 | 0.57 | 0.62 | 0.65 | 0.592 |
| Dense only (no BM25) + rerank | 0.56 | 0.60 | 0.62 | 0.578 |

Down-weighting or dropping BM25 both lose recall versus equal-weight RRF, so the shipped default is weight 1.0.
Best R@10 belongs to dense chunk-level alone (0.76) because it ranks all 68k passages directly instead of
drawing the top-10 from a rerank pool; the shipped pipeline optimises R@1/MRR, which is what a 5-result UI shows.
