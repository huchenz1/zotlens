"""MCP server exposing ZotLens retrieval to any MCP client (Claude Desktop, Cursor, WorkBuddy, ...).

Run (stdio):  python -m zotlens.mcp_server
Run (HTTP):   python -m zotlens.mcp_server --http   -> http://127.0.0.1:8766/mcp
"""
from __future__ import annotations

import sys

try:  # mcp >= 2.0
    from mcp.server.mcpserver import MCPServer as FastMCP
except ImportError:  # mcp 1.x
    from mcp.server.fastmcp import FastMCP

from .retriever import library
from .tools import SKILLS, SourceRegistry, run_tool

mcp = FastMCP("zotlens", instructions=(
    "Search and read the user's local Zotero library (full text, hybrid retrieval + reranking). "
    "Cite papers by their [n] numbers; every result includes Zotero links and PDF page numbers."))
_REG = SourceRegistry()  # one citation numbering per MCP server process


@mcp.tool()
def search_library(query: str, keywords: list[str] | None = None, collection: str | None = None,
                   year_from: int | None = None, year_to: int | None = None, top_k: int = 6) -> str:
    """Hybrid semantic + keyword search over the full text of all papers in the local Zotero library.
    query: precise English query. keywords: optional alternative phrasings. Returns ranked papers with
    evidence passages, PDF page numbers and zotero:// links."""
    text, cards, _ = run_tool("search_library", {"query": query, "keywords": keywords, "collection": collection,
                                                 "year_from": year_from, "year_to": year_to, "top_k": top_k}, _REG, 0)
    links = "\n".join(f"[{c['n']}] {c['links'].get('open_pdf') or c['links']['select']}" for c in cards)
    return text + ("\n\nZotero links:\n" + links if links else "")


@mcp.tool()
def read_paper(paper: str, question: str, k: int = 4) -> str:
    """Return the passages of ONE paper (source number from earlier results, or Zotero item key)
    most relevant to a question, with page numbers."""
    return run_tool("read_paper", {"paper": paper, "question": question, "k": k}, _REG, 0)[0]


@mcp.tool()
def find_papers(title: str | None = None, author: str | None = None, year: int | None = None,
                collection: str | None = None, tag: str | None = None, sort: str = "dateAdded", limit: int = 12) -> str:
    """Exact metadata lookup (title words, author, year, collection, tag) or list recently added papers."""
    return run_tool("find_papers", {"title": title, "author": author, "year": year, "collection": collection,
                                    "tag": tag, "sort": sort, "limit": limit}, _REG, 0)[0]


@mcp.tool()
def list_collections() -> str:
    """List Zotero collections with paper counts."""
    return run_tool("list_collections", {}, _REG, 0)[0]


@mcp.resource("zotlens://skills/{name}")
def skill(name: str) -> str:
    """Step-by-step procedures: literature-search, compare-papers, related-work, paper-deep-read."""
    return SKILLS.get(name, {}).get("body", f"Unknown skill. Available: {list(SKILLS)}")


def _make_prompt(body: str):
    def prompt(topic: str = "") -> str:
        return f"{body}\n\nTask: {topic}"
    return prompt


for _name, _s in SKILLS.items():
    mcp.prompt(name=_name, description=_s["description"])(_make_prompt(_s["body"]))


def main():
    library()  # load index up front
    if "--http" in sys.argv:
        try:
            mcp.run(transport="streamable-http", host="127.0.0.1", port=8766)
        except TypeError:  # mcp 1.x
            mcp.settings.port = 8766
            mcp.run(transport="streamable-http")
    else:
        mcp.run()


if __name__ == "__main__":
    main()
