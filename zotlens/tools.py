"""Tool definitions shared by the chat agent (OpenAI function calling) and the MCP server."""
from __future__ import annotations

import re

from .config import ROOT
from .retriever import library

SKILLS_DIR = ROOT / "skills"


def load_skills() -> dict[str, dict]:
    out = {}
    for f in sorted(SKILLS_DIR.glob("*/SKILL.md")):
        txt = f.read_text()
        m = re.match(r"---\s*\n(.*?)\n---\s*\n(.*)", txt, re.S)
        meta = dict(re.findall(r"^(\w+):\s*(.+)$", m.group(1), re.M)) if m else {}
        out[meta.get("name", f.parent.name)] = {"description": meta.get("description", ""),
                                                "body": m.group(2).strip() if m else txt}
    return out


SKILLS = load_skills()

TOOL_SPECS = [
    {"type": "function", "function": {
        "name": "search_library",
        "description": ("Semantic + keyword hybrid search over the full text of all papers in the user's local "
                        "Zotero library. Returns ranked papers with evidence passages and PDF page numbers. "
                        "Use for any question about research findings, methods, or 'which papers ...'."),
        "parameters": {"type": "object", "properties": {
            "query": {"type": "string", "description": "Main search query, a precise English sentence or phrase."},
            "keywords": {"type": "array", "items": {"type": "string"},
                         "description": "Optional 1-3 alternative phrasings / key terms to improve recall."},
            "collection": {"type": "string", "description": "Optional Zotero collection name to restrict to."},
            "year_from": {"type": "integer"}, "year_to": {"type": "integer"},
            "top_k": {"type": "integer", "description": "Number of papers (default 6, max 12)."}},
            "required": ["query"]}}},
    {"type": "function", "function": {
        "name": "read_paper",
        "description": ("Read inside ONE paper: returns the passages of that paper most relevant to a question, "
                        "with page numbers. Use for details of a specific paper (method, results, limitations)."),
        "parameters": {"type": "object", "properties": {
            "paper": {"type": "string", "description": "Source number like '3' (from earlier results) or a Zotero item key."},
            "question": {"type": "string", "description": "What to look for, in English."},
            "k": {"type": "integer", "description": "Number of passages (default 4)."}},
            "required": ["paper", "question"]}}},
    {"type": "function", "function": {
        "name": "find_papers",
        "description": ("Exact metadata lookup in the library: by title words, author name, year, collection or tag; "
                        "or list recently added papers. Use for 'papers by X', 'the paper titled ...', 'what did I add recently'."),
        "parameters": {"type": "object", "properties": {
            "title": {"type": "string"}, "author": {"type": "string"}, "year": {"type": "integer"},
            "collection": {"type": "string"}, "tag": {"type": "string"},
            "sort": {"type": "string", "enum": ["dateAdded", "year"]},
            "limit": {"type": "integer"}}}}},
    {"type": "function", "function": {
        "name": "list_collections",
        "description": "List the collections (folders) of the user's Zotero library with paper counts.",
        "parameters": {"type": "object", "properties": {}}}},
    {"type": "function", "function": {
        "name": "load_skill",
        "description": "Load the step-by-step instructions of an available skill before performing that kind of task.",
        "parameters": {"type": "object", "properties": {
            "name": {"type": "string", "enum": list(SKILLS)}}, "required": ["name"]}}},
]


class SourceRegistry:
    """Assigns stable citation numbers [n] to papers across a conversation."""

    def __init__(self):
        self.by_key: dict[str, dict] = {}
        self.order: list[str] = []

    def add(self, paper: dict, evidence: list[dict], turn: int, query: str | None = None) -> dict:
        k = paper["key"]
        if k not in self.by_key:
            self.order.append(k)
            self.by_key[k] = {**paper, "n": len(self.order), "evidence": [], "turns": [], "queries": []}
        s = self.by_key[k]
        for f in ("relevance", "score", "links"):
            if paper.get(f) is not None:
                s[f] = paper[f]
        seen = {(e.get("chunk_id"), e.get("kind")) for e in s["evidence"]}
        for e in evidence:
            if (e.get("chunk_id"), e.get("kind")) not in seen:
                s["evidence"].append(e)
        s["evidence"].sort(key=lambda e: -(e.get("score") or -99))
        s["evidence"] = s["evidence"][:4]
        if turn not in s["turns"]:
            s["turns"].append(turn)
        if query and query not in s["queries"]:
            s["queries"].append(query)
        s["last_turn"] = turn
        return s

    def resolve(self, ref: str) -> str | None:
        ref = str(ref).strip().strip("[]")
        if ref.isdigit() and 1 <= int(ref) <= len(self.order):
            return self.order[int(ref) - 1]
        return ref or None

    def all(self) -> list[dict]:
        return [self.by_key[k] for k in self.order]


def _short_auth(a: list[str]) -> str:
    if not a:
        return "n/a"
    return a[0] + (" et al." if len(a) > 2 else (" & " + a[1] if len(a) == 2 else ""))


def _clip(t: str, n: int) -> str:
    w = t.split()
    return " ".join(w[:n]) + (" ..." if len(w) > n else "")


def run_tool(name: str, args: dict, reg: SourceRegistry, turn: int) -> tuple[str, list[dict], dict]:
    """Execute a tool. Returns (text for the LLM, source cards touched, ui meta)."""
    lib = library()
    if name == "search_library":
        top_k = max(1, min(int(args.get("top_k") or 6), 10))
        r = lib.search(args["query"], args.get("keywords") or [], args.get("collection"),
                       args.get("year_from"), args.get("year_to"), top_k=top_k)
        cards, lines = [], []
        for x in r["results"]:
            s = reg.add(x, x["evidence"], turn, args["query"])
            cards.append(s)
            ev = " | ".join(
                f"({'abstract' if e['kind']=='abstract' else 'p.'+str(e['page']) if e['page'] else 'passage'}) {_clip(e['text'], 90)}"
                for e in x["evidence"])
            lines.append(f"[{s['n']}] {x['title']} — {_short_auth(x['authors'])}, {x['year'] or 'n.d.'}"
                         f"{', ' + x['venue'] if x['venue'] else ''}; "
                         f"relevance={'n/a' if x['relevance'] is None else format(x['relevance'], '.2f')}; "
                         f"collections={x['collections']}\n    evidence: {ev}")
        text = (f"{len(cards)} papers for '{args['query']}' (search took {r['timing']['total_s']}s):\n" + "\n".join(lines)) \
            if cards else "No matching papers in the library."
        return text, cards, {"timing": r["timing"], "coverage": r["coverage"], "keywords": r["keywords"]}

    if name == "read_paper":
        key = reg.resolve(args["paper"])
        r = lib.paper_passages(key, args["question"], int(args.get("k") or 4)) if key else None
        if not r:
            return f"Paper '{args['paper']}' not found.", [], {}
        s = reg.add(r, r["passages"], turn)
        body = "\n".join(f"- (p.{e['page'] or '?'}) {_clip(e['text'], 170)}" for e in r["passages"])
        return (f"[{s['n']}] {r['title']} ({r['year']}). Abstract: {_clip(r['abstract'], 120)}\n"
                f"Relevant passages for '{args['question']}':\n{body}"), [s], {}

    if name == "find_papers":
        res = lib.search_metadata(args.get("title"), args.get("author"), args.get("year"), args.get("collection"),
                                  args.get("tag"), int(args.get("limit") or 12), args.get("sort") or "dateAdded")
        cards = [reg.add(p, [{"text": p["abstract"], "score": None, "chunk_id": None, "page": None, "kind": "abstract"}]
                         if p["abstract"] else [], turn) for p in res]
        lines = [f"[{c['n']}] {c['title']} — {_short_auth(c['authors'])}, {c['year'] or 'n.d.'}; collections={c['collections']}"
                 for c in cards]
        return (f"{len(cards)} papers:\n" + "\n".join(lines)) if cards else "No papers match.", cards, {}

    if name == "list_collections":
        return "Collections: " + ", ".join(f"{c['name']} ({c['count']})" for c in lib.collections) + \
            f". Total papers: {len(lib.papers)}.", [], {}

    if name == "load_skill":
        sk = SKILLS.get(args.get("name", ""))
        return (sk["body"] if sk else f"Unknown skill. Available: {list(SKILLS)}"), [], {"skill": args.get("name")}

    return f"Unknown tool {name}", [], {}
