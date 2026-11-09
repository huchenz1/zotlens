"""Tool-calling chat agent with streaming events.

Events yielded to the UI:
  {"type": "status", "text"}                       what the agent is doing
  {"type": "tool", "id", "name", "args", "state": "start"|"end", "summary", "meta"}
  {"type": "sources", "cards": [...]}              new/updated source cards (citation numbers)
  {"type": "thinking", "text"}                     reasoning tokens (local thinking models)
  {"type": "token", "text"}                        answer tokens
  {"type": "done", "cited": [n...], "usage"}
  {"type": "error", "text"}
"""
from __future__ import annotations

import json
import re
import time
import uuid

import httpx
from openai import OpenAI

from .config import llm_config
from .retriever import library
from .tools import SKILLS, TOOL_SPECS, SourceRegistry, run_tool

MAX_STEPS = 6
MAX_STEP_SECONDS = 300   # hard wall-clock guard per LLM call, so a stalled stream cannot hang the UI forever


def system_prompt() -> str:
    lib = library()
    colls = ", ".join(f"{c['name']} ({c['count']})" for c in lib.collections)
    skills = "\n".join(f"- {n}: {s['description']}" for n, s in SKILLS.items())
    return f"""You are ZotLens, a research assistant connected to the user's local Zotero library
({len(lib.papers)} papers; collections: {colls}). Today is {time.strftime('%Y-%m-%d')}.

You can chat about anything. Decide yourself whether the library is needed:
- Questions about research topics, findings, methods, "which papers", comparisons, related work,
  or anything the user's papers could answer -> use the tools first, then answer from the evidence.
- Small talk, general programming / math / writing help that does not need papers -> just answer, no tools.

Rules for library-grounded answers:
- Cite papers with their source numbers in square brackets, e.g. [3] or [2][5], right after the claim.
- Only cite numbers returned by tools in this conversation. Never invent papers, authors, numbers or page numbers.
- Base claims on the returned evidence; if evidence is thin, say so. If the library lacks relevant papers,
  say that clearly; you may add general knowledge but mark it as "(not from your library)".
- Search queries should be in English even if the user writes Chinese; answer in the user's language.
- Be concise and structured (short paragraphs, bullets, or a table for comparisons).
- Efficiency: one search_library call (with good keywords) is usually enough; search again only if results are
  weak. Use read_paper only when the user asks about specific papers' details or the task needs it
  (compare / deep read), at most 3 read_paper calls per user turn. Issue independent tool calls in parallel.

Skills (call load_skill to get the step-by-step procedure before doing these tasks):
{skills}
"""


def _client(cfg):
    return OpenAI(base_url=cfg["base_url"], api_key=cfg["api_key"] or "none",
                  http_client=httpx.Client(trust_env=False, timeout=120))


def resolve_model(cfg) -> str:
    if cfg["model"]:
        return cfg["model"]
    r = httpx.get(cfg["base_url"].rstrip("/") + "/models", timeout=5, trust_env=False)
    return r.json()["data"][0]["id"]


class Session:
    def __init__(self, sid: str | None = None):
        self.id = sid or uuid.uuid4().hex[:12]
        self.messages: list[dict] = []
        self.sources = SourceRegistry()
        self.turn = 0
        self.created = time.time()


SESSIONS: dict[str, Session] = {}


def get_session(sid: str | None) -> Session:
    if sid and sid in SESSIONS:
        return SESSIONS[sid]
    s = Session(sid)
    SESSIONS[s.id] = s
    return s


def _tool_summary(name, args, cards, meta):
    if name == "search_library":
        return f"找到 {len(cards)} 篇 · {meta.get('timing', {}).get('total_s', '?')}s"
    if name == "read_paper":
        return f"读取 [{cards[0]['n']}] 的相关段落" if cards else "未找到该论文"
    if name == "find_papers":
        return f"匹配 {len(cards)} 篇"
    if name == "load_skill":
        return f"已加载技能 {args.get('name')}"
    return "完成"


def chat(session: Session, user_text: str, attached: list[int] | None = None):
    cfg = llm_config()
    try:
        client = _client(cfg)
        model = resolve_model(cfg)
    except Exception as e:  # noqa: BLE001
        yield {"type": "error", "text": f"LLM 后端不可用 ({cfg['backend']}): {e}"}
        return

    session.turn += 1
    turn = session.turn
    content = user_text
    if attached:
        refs = ", ".join(f"[{n}]" for n in attached)
        content += f"\n\n(The user selected sources {refs} in the side panel; they refer to these papers.)"
    session.messages.append({"role": "user", "content": content})

    usage_total = {"prompt_tokens": 0, "completion_tokens": 0}
    for step in range(MAX_STEPS):
        msgs = [{"role": "system", "content": system_prompt()}] + session.messages
        try:
            stream = client.chat.completions.create(
                model=model, messages=msgs, tools=TOOL_SPECS, stream=True, temperature=0.3,
                max_tokens=4000, stream_options={"include_usage": True}, extra_body=cfg["extra_body"] or None)
        except Exception as e:  # noqa: BLE001
            yield {"type": "error", "text": f"LLM 调用失败: {e}"}
            return

        text, calls = "", {}
        deadline = time.time() + MAX_STEP_SECONDS
        try:
            for chunk in stream:
                if time.time() > deadline:
                    yield {"type": "error", "text": f"模型响应超时（>{MAX_STEP_SECONDS}s），已中断本轮。可以重试或换一个更短的问题。"}
                    return
                if getattr(chunk, "usage", None):
                    usage_total["prompt_tokens"] += chunk.usage.prompt_tokens or 0
                    usage_total["completion_tokens"] += chunk.usage.completion_tokens or 0
                if not chunk.choices:
                    continue
                d = chunk.choices[0].delta
                rc = getattr(d, "reasoning_content", None)
                if rc:
                    yield {"type": "thinking", "text": rc}
                if d.content:
                    text += d.content
                    yield {"type": "token", "text": d.content}
                for tc in d.tool_calls or []:
                    c = calls.setdefault(tc.index, {"id": "", "name": "", "args": ""})
                    if tc.id:
                        c["id"] = tc.id
                    if tc.function and tc.function.name:
                        c["name"] += tc.function.name
                    if tc.function and tc.function.arguments:
                        c["args"] += tc.function.arguments
        except Exception as e:  # noqa: BLE001
            yield {"type": "error", "text": f"流式输出中断: {e}"}
            return

        if not calls:
            session.messages.append({"role": "assistant", "content": text})
            cited = sorted({int(n) for n in re.findall(r"\[(\d{1,3})\]", text)
                            if 1 <= int(n) <= len(session.sources.order)})
            yield {"type": "done", "cited": cited, "usage": usage_total, "model": model, "backend": cfg["backend"]}
            return

        ordered = [calls[i] for i in sorted(calls)]
        session.messages.append({"role": "assistant", "content": text or None, "tool_calls": [
            {"id": c["id"] or f"call_{uuid.uuid4().hex[:8]}", "type": "function",
             "function": {"name": c["name"], "arguments": c["args"] or "{}"}} for c in ordered]})
        if text:
            yield {"type": "token", "text": "\n\n"}
        for c, msg_call in zip(ordered, session.messages[-1]["tool_calls"]):
            try:
                args = json.loads(c["args"] or "{}")
            except json.JSONDecodeError:
                args = {}
            yield {"type": "tool", "id": msg_call["id"], "name": c["name"], "args": args, "state": "start"}
            try:
                result_text, cards, meta = run_tool(c["name"], args, session.sources, turn)
            except Exception as e:  # noqa: BLE001
                result_text, cards, meta = f"Tool error: {e}", [], {}
            if cards:
                yield {"type": "sources", "cards": cards, "turn": turn}
            yield {"type": "tool", "id": msg_call["id"], "name": c["name"], "args": args, "state": "end",
                   "summary": _tool_summary(c["name"], args, cards, meta), "meta": meta,
                   "refs": [x["n"] for x in cards]}
            session.messages.append({"role": "tool", "tool_call_id": msg_call["id"], "content": result_text})

    yield {"type": "error", "text": "达到最大工具调用步数，已停止。"}
