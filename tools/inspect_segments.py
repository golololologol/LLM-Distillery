"""
Dump the pipeline's internal segmented state for a dataset, for manual inspection.

Uses the real SentinelFormatter from the pipeline. For each sample:
  - shows the raw messages
  - shows the rendered text (post-template, post-splice)
  - shows every segment with byte ranges, type, hash, truncated flag,
    and the actual byte slice extracted from the rendered text
  - shows an annotated rendered text with [SEG#N type] ... [/SEG] markers
    inserted at the byte ranges, so you can eyeball whether the ranges land
    where they should

Usage:
  python -m tools.inspect_segments <dataset.jsonl> [--out report.txt]
                                   [--tokenizer test_data/tiny_tokenizer]
                                   [--context-len 4096] [--limit N]
                                   [--no-reasoning] [--no-tools]
                                   [--save-roles assistant,...]
                                   [--seg-handling content=active,reasoning=active,tool_call=active]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer

from classes.formatter import SentinelFormatter
from utils.dataset_utils import read_jsonl


def parse_handling(s: str | None) -> dict | None:
    if not s:
        return None
    out = {}
    for part in s.split(","):
        k, v = part.split("=")
        out[k.strip()] = v.strip()
    return out


def load_jsonl(path: str, limit: int | None, dataset_format: str = "auto") -> list[dict]:
    rows = read_jsonl(path, dataset_format=dataset_format)
    if limit:
        rows = rows[:limit]
    return rows


def annotate(text_bytes: bytes, segments) -> str:
    # Build an inline annotation by inserting markers at byte boundaries.
    # Sort by byte_start; insert from the back so earlier offsets stay valid.
    events = []
    for i, s in enumerate(segments):
        open_tag = f"\u27e8SEG#{i} {s.type} msg={s.msg_idx}{' TRUNC' if s.truncated else ''}\u27e9".encode("utf-8")
        close_tag = f"\u27e8/SEG#{i}\u27e9".encode("utf-8")
        events.append((s.byte_end, "close", close_tag))
        events.append((s.byte_start, "open", open_tag))
    # Insert in reverse order; ties: closes before opens at same position would
    # nest wrong, but segments shouldn't overlap, so order doesn't matter much.
    events.sort(key=lambda e: (e[0], 0 if e[1] == "close" else 1), reverse=True)
    buf = bytearray(text_bytes)
    for pos, _kind, tag in events:
        buf[pos:pos] = tag
    return buf.decode("utf-8", errors="replace")


def dump_sample(out, idx: int, sample: dict, formatter: SentinelFormatter,
                save_roles: set[str], context_len: int, seg_handling: dict | None):
    sample_id = sample.get("id", idx)
    messages = sample["messages"]

    out.write("=" * 100 + "\n")
    out.write(f"SAMPLE idx={idx} id={sample_id}\n")
    out.write("=" * 100 + "\n\n")

    out.write("--- RAW MESSAGES ---\n")
    for i, m in enumerate(messages):
        out.write(f"  [{i}] role={m.get('role')!r}\n")
        for k, v in m.items():
            if k == "role":
                continue
            preview = repr(v)
            if len(preview) > 400:
                preview = preview[:400] + "...<truncated>"
            out.write(f"        {k}: {preview}\n")
    out.write("\n")

    try:
        text, tokens, segments, was_cropped, crop_end, anchors = formatter.render(
            messages,
            save_roles=save_roles,
            context_len=context_len,
            segment_handling=seg_handling,
        )
    except Exception as e:
        out.write(f"!! render() FAILED: {type(e).__name__}: {e}\n\n")
        return

    text_bytes = text.encode("utf-8")
    out.write(f"--- RENDER RESULT ---\n")
    out.write(f"  rendered_text_bytes: {len(text_bytes)}\n")
    out.write(f"  token_count:         {len(tokens)} (context_len={context_len})\n")
    out.write(f"  was_cropped:         {was_cropped}\n")
    out.write(f"  crop_end (bytes):    {crop_end}\n")
    out.write(f"  active_segments:     {len(segments)}\n\n")

    out.write("--- SEGMENTS (active only) ---\n")
    if not segments:
        out.write("  (none)\n")
    for i, s in enumerate(segments):
        slice_bytes = text_bytes[s.byte_start:s.byte_end]
        try:
            slice_preview = slice_bytes.decode("utf-8")
        except UnicodeDecodeError:
            slice_preview = slice_bytes.decode("utf-8", errors="replace") + " <invalid utf-8>"
        if len(slice_preview) > 300:
            slice_preview = slice_preview[:300] + "...<truncated>"
        out.write(
            f"  [{i}] msg_idx={s.msg_idx} segment_idx={s.segment_idx} type={s.type} "
            f"role={s.role} handling={s.handling} "
            f"tool_call_idx={s.tool_call_idx} "
            f"bytes=[{s.byte_start},{s.byte_end}) len={s.byte_count} "
            f"truncated={s.truncated} hash={s.canonical_target_hash[:16]}\n"
        )
        out.write(f"        slice: {slice_preview!r}\n")
    out.write("\n")

    out.write("--- ANCHORS (event-channel boundary tokens, plan v8 \u00a75) ---\n")
    if not anchors:
        out.write("  (none)\n")
    for a in anchors:
        out.write(
            f"  msg_idx={a.msg_idx} segment_idx={a.segment_idx} type={a.type} "
            f"side={a.anchor_side} token_pos={a.token_pos} "
            f"unreachable={a.unreachable} truncated={a.truncated}\n"
        )
    out.write("\n")

    out.write("--- RENDERED TEXT (raw) ---\n")
    out.write(text)
    if not text.endswith("\n"):
        out.write("\n")
    out.write("--- /RENDERED TEXT ---\n\n")

    out.write("--- RENDERED TEXT (annotated with segment ranges) ---\n")
    out.write(annotate(text_bytes, segments))
    out.write("\n--- /ANNOTATED ---\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", help="Path to JSONL dataset")
    ap.add_argument("--out", default=None, help="Output report path (default: <dataset>.segments.txt)")
    ap.add_argument("--tokenizer", default=str(ROOT / "test_data" / "tiny_tokenizer"))
    ap.add_argument("--context-len", type=int, default=4096)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-reasoning", action="store_true")
    ap.add_argument("--no-tools", action="store_true")
    ap.add_argument("--save-roles", default="assistant",
                    help="comma-separated roles considered 'active' for training")
    ap.add_argument("--seg-handling", default=None,
                    help="e.g. content=active,reasoning=context,tool_call=disabled")
    ap.add_argument("--dataset-format", default="auto",
                    help="Dataset format strategy name. See classes/dataset_formats.py.")
    args = ap.parse_args()

    out_path = args.out or (args.dataset + ".segments.txt")
    save_roles = {r.strip() for r in args.save_roles.split(",") if r.strip()}
    seg_handling = parse_handling(args.seg_handling)

    print(f"Loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    chat_template = getattr(tok, "chat_template", None)
    if not chat_template:
        tpl_path = Path(args.tokenizer) / "chat_template.jinja"
        if tpl_path.exists():
            chat_template = tpl_path.read_text(encoding="utf-8")
    formatter = SentinelFormatter(
        tok,
        chat_template=chat_template,
        render_options=None,
        supports_reasoning=not args.no_reasoning,
        supports_tool_calls=not args.no_tools,
    )

    print(f"Loading dataset: {args.dataset}")
    samples = load_jsonl(args.dataset, args.limit, dataset_format=args.dataset_format)
    print(f"  {len(samples)} samples")

    print(f"Writing report: {out_path}")
    with open(out_path, "w", encoding="utf-8") as out:
        out.write(f"# Segment inspection report\n")
        out.write(f"# dataset:        {os.path.abspath(args.dataset)}\n")
        out.write(f"# tokenizer:      {args.tokenizer}\n")
        out.write(f"# context_len:    {args.context_len}\n")
        out.write(f"# save_roles:     {sorted(save_roles)}\n")
        out.write(f"# seg_handling:   {seg_handling}\n")
        out.write(f"# supports_reasoning: {formatter.supports_reasoning}\n")
        out.write(f"# supports_tool_calls: {formatter.supports_tool_calls}\n")
        out.write(f"# samples:        {len(samples)}\n\n")

        for idx, sample in enumerate(samples):
            dump_sample(out, idx, sample, formatter, save_roles, args.context_len, seg_handling)

    print("Done.")


if __name__ == "__main__":
    main()
