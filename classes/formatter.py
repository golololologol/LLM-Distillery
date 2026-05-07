from __future__ import annotations

import hashlib
import json
import re

from typing import Any

from classes.data_classes import Segment, Anchor


_SENTINEL_RE = re.compile(rb"SEG[0-9A-F]{8}[CRA]")


def _sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def hash_tool_call(tc: dict) -> str:
    payload = json.dumps(
        {"name": tc["name"], "arguments": tc["arguments"], "id": tc["id"]},
        sort_keys=True,
    )
    return _sha(payload.encode("utf-8"))


def _crop_byte_end(tokenizer, text: str, tokens: list[int], context_len: int) -> int:
    try:
        enc = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
        offsets = enc["offset_mapping"]
        enc_tokens = enc["input_ids"]
        if offsets:
            token_offset = 0
            if tokens and enc_tokens and tokens[0] != enc_tokens[0]:
                token_offset = 1
            idx = context_len - 1 - token_offset
            if 0 <= idx < len(offsets) and offsets[idx][1] > 0:
                return len(text[: offsets[idx][1]].encode("utf-8"))
    except (AttributeError, TypeError, NotImplementedError, KeyError):
        pass
    return len(tokenizer.decode(tokens[:context_len], skip_special_tokens=False).encode("utf-8"))


class SentinelFormatter:
    def __init__(
        self,
        tokenizer,
        *,
        chat_template: str | None = None,
        render_options: dict | None = None,
        supports_reasoning: bool = True,
        supports_tool_calls: bool = True,
        tokenizer_kwargs: dict | None = None,
    ):
        self.tokenizer = tokenizer
        raw_template = chat_template or getattr(tokenizer, "chat_template", None)
        # HF allows tokenizer.chat_template to be a dict mapping a template
        # name (e.g. "default", "tool_use", "rag" for Cohere) to a Jinja
        # string. Pick the default entry; fall back to the first value.
        if isinstance(raw_template, dict):
            raw_template = raw_template.get("default") or next(iter(raw_template.values()), None)
        self._chat_template = raw_template
        if not self._chat_template:
            raise ValueError(
                "SentinelFormatter: no chat_template provided and tokenizer has none"
            )
        self.render_options = dict(render_options or {})
        self.tokenizer_kwargs = tokenizer_kwargs or getattr(tokenizer, "tokenizer_kwargs", {}) or {}
        self._template_sha = hashlib.sha256(self._chat_template.encode("utf-8")).hexdigest()[:16]
        self.supports_reasoning = supports_reasoning
        self.supports_tool_calls = supports_tool_calls
        self.name = f"sentinel:{getattr(tokenizer, 'name_or_path', 'unknown')}:{self._template_sha}"
        self.version = "7"
        self._validate_template_support()
    
    def _validate_template_support(self) -> None:
        """Probe the template with a fake message and verify that declared
        segment types (content, reasoning, tool_call) are actually emitted.

        Raises ValueError with actionable messages on mismatch.
        """
        # Probe message that exercises every possible segment
        probe: list[dict[str, Any]] = [
            {"role": "user", "content": "Hi"},
            {
                "role": "assistant",
                "content": "answer",
                "reasoning": "thought",
                "tool_calls": [
                    {"id": "tc1", "name": "func", "arguments": "{}"}
                ],
            },
        ]
        # We need all relevant handling modes to 'active' for the probe
        probe_handling = {
            "content": "active",
            "reasoning": "active",
            "tool_call": "active",          # required for splitting
            "tool_name": "active",
            "tool_arguments": "active",
        }
        try:
            _, _, segments, _, _, _ = self.render(
                probe,
                save_roles={"assistant"},
                context_len=4096,
                segment_handling=probe_handling,
            )
        except Exception as e:
            raise RuntimeError(
                f"SentinelFormatter validation render failed: {e}"
            ) from e

        observed = {segment.type for segment in segments}

        # ---- critical invariants ----
        if "content" not in observed:
            raise ValueError(
                f"Chat template for tokenizer '{self.tokenizer.name_or_path}' "
                f"does not emit content segments – sentinel missing."
            )
        if self.supports_reasoning and "reasoning" not in observed:
            raise ValueError(
                f"supports_reasoning=True but the template for '{self.tokenizer.name_or_path}' "
                f"produced no reasoning segment. Set supports_reasoning=False."
            )
        if self.supports_tool_calls:
            if "tool_name" not in observed or "tool_arguments" not in observed:
                raise ValueError(
                    f"supports_tool_calls=True but the template for '{self.tokenizer.name_or_path}' "
                    f"did not emit tool_name/tool_arguments segments. Set supports_tool_calls=False."
                )

        # ---- defensive: check for duplicated non‑tool segments ----
        counts: dict[str, int] = {}
        for segment in segments:
            if segment.type not in ("tool_name", "tool_arguments"):
                counts[segment.type] = counts.get(segment.type, 0) + 1
        for typ, cnt in counts.items():
            if cnt > 1:
                raise ValueError(
                    f"Chat template for '{self.tokenizer.name_or_path}' "
                    f"emitted multiple '{typ}' segments in one message – possible sentinel duplication."
                )

    def options_fingerprint(self):
        opts = tuple(sorted(self.render_options.items()))
        flags = (self.supports_reasoning, self.supports_tool_calls)
        return (opts, self._template_sha, flags)

    def _build_probe(self, messages, save_roles, segment_handling=None):
        sentinels = []
        render_msgs = []
        counter = 0

        def mint(tag: str) -> str:
            nonlocal counter
            s = f"SEG{counter:08X}{tag}"
            counter += 1
            return s

        def mode(t: str) -> str:
            if segment_handling is None:
                return "active"
            return segment_handling.get(t, "active")

        def emit_sentinel(role, msg_idx, t, raw, hash_val, tool_call_idx=None):
            m = mode(t)
            if m == "disabled":
                return None
            tag_map = {
                "content": "C", "reasoning": "R", "tool_call": "A",
                "tool_name": "N", "tool_arguments": "G",
            }
            s = mint(tag_map[t])
            is_active = (m == "active") and (role in save_roles)
            sentinels.append({
                "sentinel": s, "msg_idx": msg_idx, "type": t,
                "tool_call_idx": tool_call_idx, "raw": raw, "hash": hash_val,
                "is_active": is_active, "role": role, "handling": m,
            })
            return s

        for i, m in enumerate(messages):
            role = m["role"]
            r = {"role": role}
            content = m.get("content")
            if content is not None and mode("content") != "disabled":
                raw = content.encode("utf-8")
                s = emit_sentinel(role, i, "content", raw, _sha(raw))
                r["content"] = s
            else:
                r["content"] = ""

            if role == "assistant":
                reasoning = m.get("reasoning")
                if reasoning is not None and self.supports_reasoning and mode("reasoning") != "disabled":
                    raw = reasoning.encode("utf-8")
                    s = emit_sentinel(role, i, "reasoning", raw, _sha(raw))
                    r["reasoning_content"] = s
                tcs = m.get("tool_calls") or []
                if tcs and self.supports_tool_calls and mode("tool_call") != "disabled":
                    out = []
                    # Plan v8 §6: emit separate sentinels for the tool name
                    # and the tool arguments so each becomes its own segment.
                    # ``tool_name`` / ``tool_arguments`` per-segment handling
                    # falls back to the umbrella ``tool_call`` mode when not
                    # explicitly configured (we apply that rule via ``mode()``
                    # below so user-provided overrides win).
                    parent_mode = mode("tool_call")

                    def _child_mode(child_type: str) -> str:
                        if segment_handling is None:
                            return parent_mode
                        return segment_handling.get(child_type, parent_mode)

                    for j, tc in enumerate(tcs):
                        name_raw = tc["name"].encode("utf-8")
                        args_raw = tc["arguments"].encode("utf-8")
                        joint_hash = hash_tool_call(tc)

                        # Resolve child-type effective handling once and
                        # reuse it through emit_sentinel by temporarily
                        # injecting it via the closure-captured ``mode``.
                        name_handling = _child_mode("tool_name")
                        args_handling = _child_mode("tool_arguments")

                        def _emit(child_type, raw, hash_val, child_handling):
                            if child_handling == "disabled":
                                return None
                            tag = {"tool_name": "N", "tool_arguments": "G"}[child_type]
                            s = mint(tag)
                            is_active = (child_handling == "active") and (role in save_roles)
                            sentinels.append({
                                "sentinel": s, "msg_idx": i, "type": child_type,
                                "tool_call_idx": j, "raw": raw, "hash": hash_val,
                                "is_active": is_active, "role": role,
                                "handling": child_handling,
                            })
                            return s

                        name_sent = _emit("tool_name", name_raw, _sha(name_raw), name_handling)
                        args_sent = _emit("tool_arguments", args_raw, joint_hash, args_handling)

                        out.append({
                            "id": tc["id"], "type": "function",
                            "function": {
                                "name": name_sent if name_sent is not None else tc["name"],
                                "arguments": args_sent if args_sent is not None else tc["arguments"],
                            },
                        })
                    r["tool_calls"] = out
            elif role == "tool" and m.get("tool_call_id"):
                r["tool_call_id"] = m["tool_call_id"]
            render_msgs.append(r)
        return sentinels, render_msgs

    def _apply_template(self, render_msgs, tokenize: bool):
        return self.tokenizer.apply_chat_template(
            render_msgs,
            tokenize=tokenize,
            add_generation_prompt=False,
            chat_template=self._chat_template,
            **self.render_options,
        )

    def render(self, messages, *, save_roles, context_len, segment_handling=None):
        sentinels, render_msgs = self._build_probe(messages, save_roles=save_roles, segment_handling=segment_handling)
        rendered = self._apply_template(render_msgs, tokenize=False)
        probe_bytes = rendered.encode("utf-8")

        located = []
        for meta in sentinels:
            needle = meta["sentinel"].encode("ascii")
            positions = []
            start = 0
            while True:
                idx = probe_bytes.find(needle, start)
                if idx < 0:
                    break
                positions.append(idx)
                start = idx + len(needle)
            if not positions:
                raise ValueError(
                    f"SentinelFormatter: sentinel {meta['sentinel']!r} for msg_idx={meta['msg_idx']} "
                    f"type={meta['type']} not found in rendered output. "
                    f"The chat template mutates or drops this field."
                )
            if len(positions) > 1:
                raise ValueError(
                    f"SentinelFormatter: sentinel {meta['sentinel']!r} for msg_idx={meta['msg_idx']} "
                    f"type={meta['type']} appears {len(positions)} times in rendered output"
                )
            located.append((meta, positions))

        placed = {m["sentinel"] for m in sentinels}
        for found in _SENTINEL_RE.findall(probe_bytes):
            s = found.decode("ascii")
            if s not in placed:
                raise ValueError(
                    f"SentinelFormatter: unexpected sentinel-shaped substring {s!r} in rendered output"
                )

        splices = []
        primaries = []
        for meta, positions in located:
            sent_len = len(meta["sentinel"])
            primaries.append((meta, positions[0]))
            for p in positions:
                splices.append((p, p + sent_len, meta["raw"]))

        splices_asc = sorted(splices, key=lambda x: x[0])
        final = bytearray(probe_bytes)
        for start, end, raw in sorted(splices_asc, key=lambda x: x[0], reverse=True):
            final[start:end] = raw
        final_bytes = bytes(final)
        final_text = final_bytes.decode("utf-8")

        def probe_to_final(pos: int) -> int:
            delta = 0
            for s, e, raw in splices_asc:
                if s < pos:
                    delta += len(raw) - (e - s)
                else:
                    break
            return pos + delta

        enc = self.tokenizer(final_text, add_special_tokens=False, return_offsets_mapping=True)
        token_ids = list(enc["input_ids"])
        offsets = enc.get("offset_mapping") or []

        was_cropped = False
        crop_end = len(final_bytes)
        if len(token_ids) > context_len:
            crop_end = min(_crop_byte_end(self.tokenizer, final_text, token_ids, context_len), len(final_bytes))
            token_ids = token_ids[:context_len]
            offsets = list(offsets)[:context_len]
            final_text = final_bytes[:crop_end].decode("utf-8", errors="replace")
            was_cropped = True

        # Build a char-offset -> token-index lookup for anchor computation
        # (plan v8 §5). Offsets are character-based; we convert via the prefix
        # byte length so anchors line up with byte positions in ``final_bytes``.
        # Pre‑compute cumulative byte offsets per character (O(total_chars), once)
        text_chars = final_text
        cum_bytes = [0]
        for ch in text_chars:
            cum_bytes.append(cum_bytes[-1] + len(ch.encode("utf-8")))

        token_byte_starts = []
        token_byte_ends = []
        for s_char, e_char in offsets:
            # Guard against off‑by‑one from tokenizer edge cases
            s_char = max(0, min(s_char, len(cum_bytes) - 1))
            e_char = max(0, min(e_char, len(cum_bytes) - 1))
            token_byte_starts.append(cum_bytes[s_char])
            token_byte_ends.append(cum_bytes[e_char])

        T = len(token_ids)

        def _token_containing(byte_offset: int) -> int:
            """Index of the token whose byte span covers ``byte_offset``.

            Falls back to a linear scan; ``offsets`` is monotonically
            non-decreasing, so this could be a binary search but token counts
            here are small.
            """
            if not token_byte_starts:
                return -1
            lo, hi = 0, T - 1
            while lo < hi:
                mid = (lo + hi) // 2
                if token_byte_ends[mid] <= byte_offset:
                    lo = mid + 1
                else:
                    hi = mid
            return lo

        segments = []
        anchors_out: list[Anchor] = []
        for meta, primary_start in primaries:
            if not meta["is_active"]:
                continue
            fs = probe_to_final(primary_start)
            fe = fs + len(meta["raw"])
            if fs >= crop_end:
                continue
            truncated = False
            if fe > crop_end:
                fe = crop_end
                truncated = True
            seg = Segment(
                msg_idx=meta["msg_idx"],
                type=meta["type"],
                byte_start=fs,
                byte_end=fe,
                canonical_target_hash=meta["hash"],
                truncated=truncated,
                tool_call_idx=meta.get("tool_call_idx"),
                role=meta.get("role", "assistant"),
                handling=meta.get("handling", "active"),
            )
            segments.append(seg)

        segments.sort(key=lambda s: s.byte_start)
        # Assign segment_idx per (msg_idx) ordinal across the message.
        per_msg_counter: dict[int, int] = {}
        for s in segments:
            idx = per_msg_counter.get(s.msg_idx, 0)
            s.segment_idx = idx
            per_msg_counter[s.msg_idx] = idx + 1

        # Compute pre/post anchor token positions (plan §5). Both are stored
        # whenever they fall in ``[0, T-1]``; otherwise ``unreachable=True``.
        #
        # Convention (logits[p] predicts token[p+1]):
        #   pre_token  = (first_segment_token - 1)  -> logits there predict the
        #                segment's first token (decision to ENTER the segment).
        #   post_token = (last_segment_token)       -> logits there predict the
        #                first token AFTER the segment (decision to EXIT, e.g.
        #                end-of-turn / EOS at the close of assistant content).
        #
        # An earlier revision used ``+ 1`` for ``post_token`` (matching the
        # literal formula in plan v8 §5), but that placed the anchor *on* the
        # boundary token itself; logits there predict the *second* token after
        # the segment, which made E_END_TURN mass at end-of-assistant turns
        # collapse to ~0 instead of being naturally high. The corrected anchor
        # below restores the symmetry between pre/post.
        if T > 0 and token_byte_starts:
            for s in segments:
                pre_token = _token_containing(s.byte_start) - 1
                post_token = _token_containing(max(s.byte_end - 1, s.byte_start))
                pre_unreach = (pre_token < 0) or (pre_token >= T)
                post_unreach = (post_token < 0) or (post_token >= T) or s.truncated
                anchors_out.append(Anchor(
                    msg_idx=s.msg_idx, segment_idx=s.segment_idx, type=s.type,
                    tool_call_idx=s.tool_call_idx, anchor_side="pre",
                    token_pos=pre_token, unreachable=pre_unreach,
                    truncated=s.truncated,
                ))
                anchors_out.append(Anchor(
                    msg_idx=s.msg_idx, segment_idx=s.segment_idx, type=s.type,
                    tool_call_idx=s.tool_call_idx, anchor_side="post",
                    token_pos=post_token, unreachable=post_unreach,
                    truncated=s.truncated,
                ))

        return final_text, token_ids, segments, was_cropped, crop_end, anchors_out

    def round_trip_check(self) -> None:
        probe: list[dict[str, Any]] = [
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "world"},
        ]
        if self.supports_reasoning:
            probe[1]["reasoning"] = "thinking"
        if self.supports_tool_calls:
            probe[1]["tool_calls"] = [{"id": "1", "name": "noop", "arguments": "{}"}]
        text, _, segs, _, _, _ = self.render(probe, save_roles={"assistant"}, context_len=4096)
        tb = text.encode("utf-8")
        expected = {
            "content": b"world",
            "reasoning": b"thinking",
            "tool_name": b"noop",
            "tool_arguments": b"{}",
        }
        for s in segs:
            slice_b = tb[s.byte_start:s.byte_end]
            if slice_b != expected[s.type]:
                raise RuntimeError(
                    f"SentinelFormatter round_trip_check failed: {s.type} "
                    f"slice={slice_b!r} expected={expected[s.type]!r}"
                )
