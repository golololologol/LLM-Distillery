"""
Named-strategy registry for dataset format conversion.

Each strategy is a small function: raw_sample (dict) -> canonical_sample (dict)
where canonical_sample has a "messages" list of {role, content[, reasoning, tool_calls, tool_call_id]}.
Add new strategies by writing a function and decorating it with @register("name").

To pick a strategy, set `dataset_format = "<name>"` in pipeline config. Default "auto"
sniffs from row keys (handles openai/sharegpt/alpaca/completion).
"""
from __future__ import annotations

import json
import re
from typing import Callable


REGISTRY: dict[str, Callable[[dict], dict]] = {}


def register(name: str):
    def deco(fn):
        REGISTRY[name] = fn
        return fn
    return deco


def normalize(sample: dict, format_name: str) -> dict:
    if format_name == "auto":
        format_name = _detect(sample)
    fn = REGISTRY.get(format_name)
    if fn is None:
        raise ValueError(f"Unknown dataset_format {format_name!r}. Available: {sorted(REGISTRY)}")
    out = fn(sample)
    if "id" not in out and "id" in sample:
        out["id"] = sample["id"]
    return out


def _detect(sample: dict) -> str:
    if "messages" in sample and isinstance(sample["messages"], list) \
            and sample["messages"] and "role" in sample["messages"][0]:
        # Check for per-block roles (reasoning, tool_call, tool_output).
        roles = {m.get("role") for m in sample["messages"]}
        if roles & {"reasoning", "tool_call", "tool_output", "answer"}:
            return "per_block_roles"
        return "openai"
    if "conversations" in sample and isinstance(sample["conversations"], list):
        # Check for sharegpt with inline tool calls.
        vals = [t.get("value", "") for t in sample["conversations"]]
        if any("<tool_call>" in v for v in vals):
            return "hermes_inline"
        return "sharegpt"
    if "instruction" in sample and "output" in sample:
        return "alpaca"
    if "instruction" in sample and "response" in sample:
        return "instruction_response"
    if "query" in sample and "answer" in sample:
        return "query_answer"
    if "context" in sample and "response" in sample:
        return "dolly"
    if "text" in sample:
        return "completion"
    if "input" in sample and "output" in sample:
        return "nemotron_post_training"
    raise ValueError(f"Could not auto-detect dataset format. Keys: {sorted(sample.keys())}")


# ---------------- standard formats ----------------

@register("openai")
def openai(s):
    return {"messages": s["messages"]}


_SHAREGPT_ROLES = {"human": "user", "gpt": "assistant", "system": "system", "tool": "tool"}

@register("sharegpt")
def sharegpt(s):
    msgs = []
    if s.get("init"):
        msgs.append({"role": "system", "content": s["init"]})
    for t in s["conversations"]:
        msgs.append({"role": _SHAREGPT_ROLES.get(t["from"], t["from"]), "content": t["value"]})
    return {"messages": msgs}


@register("alpaca")
def alpaca(s):
    msgs = []
    if s.get("system"):
        msgs.append({"role": "system", "content": s["system"]})
    user = s["instruction"] + ("\n" + s["input"] if s.get("input") else "")
    msgs.append({"role": "user", "content": user})
    msgs.append({"role": "assistant", "content": s["output"]})
    return {"messages": msgs}


@register("completion")
def completion(s):
    return {"messages": [{"role": "assistant", "content": s["text"]}]}


@register("instruction_response")
def instruction_response(s):
    msgs = []
    if s.get("system"):
        msgs.append({"role": "system", "content": s["system"]})
    msgs.append({"role": "user", "content": s["instruction"]})
    msgs.append({"role": "assistant", "content": s["response"]})
    return {"messages": msgs}


@register("query_answer")
def query_answer(s):
    return {"messages": [
        {"role": "user", "content": s["query"]},
        {"role": "assistant", "content": s["answer"]},
    ]}


@register("dolly")
def dolly(s):
    instruction = s["instruction"].strip()
    context = (s.get("context") or "").strip()
    user = (context + "\n" + instruction) if context else instruction
    return {"messages": [
        {"role": "user", "content": user},
        {"role": "assistant", "content": s["response"]},
    ]}


@register("prompt_completion")
def prompt_completion(s):
    return {"messages": [
        {"role": "user", "content": s["prompt"]},
        {"role": "assistant", "content": s["completion"]},
    ]}


# ---------------- agentic / structured formats ----------------

_TOOL_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*</tool_call>", re.DOTALL)
_TOOL_RESPONSE_RE = re.compile(r"<tool_response>\s*(.*?)\s*</tool_response>", re.DOTALL)
_THINK_RE = re.compile(r"<think>\s*(.*?)\s*</think>", re.DOTALL)


@register("hermes_inline")
def hermes_inline(s):
    """Hermes/Glaive-style: <tool_call>{json}</tool_call> blocks inside content strings."""
    raw = s.get("messages") or [{"role": _SHAREGPT_ROLES.get(t["from"], t["from"]), "content": t["value"]}
                                 for t in s.get("conversations", [])]
    sid = s.get("id", 0)
    out = []
    pending_ids: list[str] = []
    for mi, m in enumerate(raw):
        role, content = m["role"], m.get("content", "") or ""
        if role == "assistant":
            calls = _extract_tool_calls(content, sid, mi)
            if calls:
                pending_ids = [c["id"] for c in calls]
                out.append({"role": "assistant", "content": _TOOL_CALL_RE.sub("", content).strip(),
                            "tool_calls": calls})
            else:
                pending_ids = []
                out.append({"role": "assistant", "content": content})
        elif role == "tool":
            for ri, body in enumerate(_TOOL_RESPONSE_RE.findall(content) or [content.strip()]):
                cid = pending_ids[ri] if ri < len(pending_ids) else f"call_{sid}_{mi}_{ri}"
                out.append({"role": "tool", "tool_call_id": cid, "content": body.strip()})
        else:
            out.append({"role": role, "content": content})
    return {"messages": out}


def _extract_tool_calls(content, sid, mi):
    calls = []
    for ci, m in enumerate(_TOOL_CALL_RE.finditer(content)):
        try:
            obj = json.loads(m.group(1))
        except json.JSONDecodeError:
            continue
        args = obj.get("arguments", "")
        if not isinstance(args, str):
            args = json.dumps(args, ensure_ascii=False)
        calls.append({"id": f"call_{sid}_{mi}_{ci}", "name": obj.get("name", ""), "arguments": args})
    return calls


@register("per_block_roles")
def per_block_roles(s):
    """Conversations split into role=reasoning|tool_call|tool_output|answer messages."""
    sid = s.get("id", 0)
    out = []
    buf: dict = {}
    asst_idx = 0
    pending_ids: list[str] = []

    def flush():
        nonlocal buf, asst_idx, pending_ids
        if not buf:
            return
        msg = {"role": "assistant", "content": buf.get("content", "")}
        if buf.get("reasoning"):
            msg["reasoning"] = buf["reasoning"]
        if buf.get("tool_calls"):
            msg["tool_calls"] = buf["tool_calls"]
            pending_ids = [tc["id"] for tc in buf["tool_calls"]]
        out.append(msg)
        asst_idx += 1
        buf = {}

    for m in s["messages"]:
        role, content = m["role"], (m.get("content") or "")
        if role in ("system", "user"):
            flush(); pending_ids = []
            out.append({"role": role, "content": content})
        elif role == "reasoning":
            mt = _THINK_RE.search(content)
            buf["reasoning"] = (mt.group(1) if mt else content).strip()
        elif role == "tool_call":
            mt = _TOOL_CALL_RE.search(content)
            try:
                obj = json.loads(mt.group(1) if mt else content)
            except (json.JSONDecodeError, AttributeError):
                continue
            args = obj.get("arguments", "")
            if not isinstance(args, str):
                args = json.dumps(args, ensure_ascii=False)
            tcs = buf.setdefault("tool_calls", [])
            tcs.append({"id": f"call_{sid}_{asst_idx}_{len(tcs)}",
                        "name": obj.get("name", ""), "arguments": args})
        elif role in ("answer", "assistant"):
            buf["content"] = content.strip()
            flush()
        elif role == "tool_output":
            if buf:
                flush()
            mt = _TOOL_RESPONSE_RE.search(content)
            body = (mt.group(1) if mt else content).strip()
            cid = pending_ids.pop(0) if pending_ids else f"call_{sid}_{asst_idx}_x"
            out.append({"role": "tool", "tool_call_id": cid, "content": body})
        else:
            flush(); pending_ids = []
            out.append({"role": role, "content": content})
    flush()
    return {"messages": out}


@register("nemotron_post_training")
def nemotron_post_training(s):
    """nvidia/Llama-Nemotron-Post-Training-Dataset: input list + output + reasoning fields."""
    msgs = list(s.get("input", []))
    if s.get("system_prompt") and not (msgs and msgs[0].get("role") == "system"):
        msgs.insert(0, {"role": "system", "content": s["system_prompt"]})
    asst = {"role": "assistant", "content": s.get("output", "")}
    reasoning = s.get("reasoning")
    if reasoning and reasoning not in ("off", "false", "no"):
        asst["reasoning"] = reasoning if isinstance(reasoning, str) else str(reasoning)
    msgs.append(asst)
    return {"messages": msgs}
