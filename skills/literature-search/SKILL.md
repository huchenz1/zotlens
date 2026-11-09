---
name: literature-search
description: Multi-round strategy for finding the right papers in the user's Zotero library for a research question (query rewriting, filters, re-search when evidence is weak).
---
# Literature search

1. Rewrite the user's question into one precise English search query (most papers are English) and 1-3 alternative phrasings or key terms (synonyms, method names, acronyms). Put the main query in `query`, the rest in `keywords`.
2. Extract filters only when the user states them explicitly: collection name, year range.
3. Call `search_library`. Inspect relevance: results with relevance < 0.5 are weak.
4. If fewer than 3 results have relevance >= 0.5, or results miss an aspect of the question, search again with a different angle (narrower method name, broader concept, or a sub-question). At most 3 search rounds.
5. Answer by grouping papers by sub-topic/idea, one or two sentences each, grounded in the returned evidence, citing [n]. State clearly if the library has little on the topic.
