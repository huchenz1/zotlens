---
name: compare-papers
description: Build a structured comparison (problem, method, data/setting, key results, limitations) across 2-6 papers already in the source list or found by search.
---
# Compare papers

1. Identify the papers to compare (source numbers the user gave, or search first).
2. For each paper call `read_paper` with a question like "main method, experimental setting, key quantitative results and limitations".
3. Output a Markdown table with columns: Paper [n] | Problem | Core idea / method | Setting / data | Key results | Limitations. Keep each cell short; only use facts present in the passages; write "not stated" otherwise.
4. Below the table, give 2-4 bullet takeaways: where they agree, where they differ, which to use when.
