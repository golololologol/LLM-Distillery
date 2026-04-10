import numpy as np
import os
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from tqdm import tqdm
from classes.data_classes import ConvoProcessed
from utils.dataset_utils import compute_content_sha

_worker_tokenizer = None

def _init_worker(model_path, chat_template=None):
    global _worker_tokenizer
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    from transformers import AutoTokenizer
    _worker_tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    if chat_template:
        _worker_tokenizer.chat_template = chat_template


def find_content_byte_ranges(
    messages: list[dict],
    tokenizer,
    save_roles: set[str],
    **template_kwargs,
) -> list[tuple[int, int]]:
    """Find byte offsets of content via incremental template rendering.

    Uses a dummy placeholder to ensure templates always render the message
    structure, then forward+backward scans to locate the actual content.
    """
    ranges = []
    for i, msg in enumerate(messages):
        if msg["role"] not in save_roles:
            continue
        content = msg.get("content", "")
        if not content:
            continue

        # Use a dummy placeholder instead of empty string - some templates
        # skip rendering entirely when content is empty.
        dummy_msg = {k: ("\x00" if k == "content" else v) for k, v in msg.items()}
        rendered_dummy = tokenizer.apply_chat_template(
            messages[:i] + [dummy_msg], tokenize=False, add_generation_prompt=False, **template_kwargs
        ).encode("utf-8")
        rendered_full = tokenizer.apply_chat_template(
            messages[:i+1], tokenize=False, add_generation_prompt=False, **template_kwargs
        ).encode("utf-8")

        # Forward scan: find where dummy and actual content diverge
        content_start = 0
        for j in range(min(len(rendered_dummy), len(rendered_full))):
            if rendered_dummy[j] != rendered_full[j]:
                content_start = j
                break

        # Backward scan: find where they reconverge (shared closing marker)
        content_end = len(rendered_full)
        for j in range(1, min(len(rendered_dummy), len(rendered_full)) + 1):
            if rendered_dummy[-j] != rendered_full[-j]:
                content_end = len(rendered_full) - j + 1
                break

        ranges.append((content_start, content_end))
    return ranges


def _merge_reasoning(messages: list[dict], reasoning_mode: str) -> list[dict]:
    """Merge reasoning fields into content for assistant messages, return new list."""
    result = []
    for msg in messages:
        msg = dict(msg)  # shallow copy
        reasoning = msg.pop("reasoning", None)
        if msg["role"] == "assistant" and reasoning:
            if reasoning_mode == "inline":
                msg["content"] = f"<think>{reasoning}</think>" + msg.get("content", "")
            elif reasoning_mode == "field":
                msg["reasoning_content"] = reasoning
        result.append(msg)
    return result


def _process_one_sample(args) -> ConvoProcessed:
    raw, idx, context_len, save_roles, include_reasoning, reasoning_mode, chat_template_kwargs, pad_token_id = args
    tokenizer = _worker_tokenizer

    sample_id = raw.get("id", idx)
    messages = raw["messages"]

    if include_reasoning:
        messages = _merge_reasoning(messages, reasoning_mode)
    else:
        messages = [{k: v for k, v in msg.items() if k != "reasoning"} for msg in messages]

    formatted_text = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, **chat_template_kwargs
    )
    tokens = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False, **chat_template_kwargs
    )
    if hasattr(tokens, "input_ids"):
        tokens = tokens.input_ids
    if not isinstance(tokens, list):
        tokens = list(tokens)

    content_byte_ranges = find_content_byte_ranges(messages, tokenizer, save_roles, **chat_template_kwargs)
    content_sha = compute_content_sha(messages, save_roles)

    length = len(tokens)
    cropped = False
    if length > context_len:
        tokens = tokens[:context_len]
        cropped = True
        length = context_len
        truncated_text = tokenizer.decode(tokens, skip_special_tokens=False)
        max_byte = len(truncated_text.encode("utf-8"))
        content_byte_ranges = [
            (s, min(e, max_byte)) for s, e in content_byte_ranges if s < max_byte
        ]

    padding = context_len - len(tokens)
    if padding > 0:
        tokens = tokens + [pad_token_id] * padding

    text_bytes = formatted_text.encode("utf-8")
    ab = bytearray()
    for s, e in content_byte_ranges:
        ab.extend(text_bytes[s:e])

    return ConvoProcessed(
        origin_convo_id=sample_id,
        tokens=np.array(tokens, dtype=np.int32),
        content_byte_ranges=content_byte_ranges,
        content_sha=content_sha,
        formatted_text=formatted_text,
        padding=padding,
        cropped=cropped,
        length=length,
        actual_bytes=np.frombuffer(ab, dtype=np.uint8).copy(),
    )


import multiprocessing

# fork has near-zero cold start; spawn pays ~9s for torch imports in workers.
# Threads: ~1.2x from GIL-releasing tokenizer ops, zero startup.
# Processes beat threads past ~7k samples (spawn) due to cold start amortization.
_PROCESS_THRESHOLD = 500 if multiprocessing.get_start_method() == "fork" else 7000


def _process_one_sample_lazy(model_path, chat_template, args_tuple):
    if _worker_tokenizer is None:
        _init_worker(model_path, chat_template)
    return _process_one_sample(args_tuple)


def preprocess_samples(
    raw_samples: list[dict],
    tokenizer,
    context_len: int,
    save_roles: set[str] = None,
    include_reasoning: bool = False,
    reasoning_mode: str = "inline",
    chat_template_kwargs: dict = None,
) -> list[ConvoProcessed]:
    if save_roles is None:
        save_roles = {"assistant"}
    if chat_template_kwargs is None:
        chat_template_kwargs = {}

    pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    args = [
        (raw, idx, context_len, save_roles, include_reasoning,
         reasoning_mode, chat_template_kwargs, pad_token_id)
        for idx, raw in enumerate(raw_samples)
    ]

    n_samples = len(raw_samples)
    n_workers = min(cpu_count(), n_samples)
    prefer = "processes" if n_samples >= _PROCESS_THRESHOLD else "threads"

    global _worker_tokenizer
    _worker_tokenizer = tokenizer
    model_path = tokenizer.name_or_path
    chat_template = getattr(tokenizer, 'chat_template', None)

    gen = Parallel(
        n_jobs=n_workers, batch_size="auto", prefer=prefer,
        return_as="generator_unordered",
    )(delayed(_process_one_sample_lazy)(model_path, chat_template, a) for a in args)

    results = list(tqdm(gen, total=n_samples, desc="  Preprocessing", unit="samples", leave=False, smoothing=0.06))
    filtered = [r for r in results if r.content_byte_ranges]
    if len(filtered) < len(results):
        dropped_ids = [r.origin_convo_id for r in results if not r.content_byte_ranges]
        print(f"  Filtered out {len(results) - len(filtered)} samples (IDs: {dropped_ids}) with no content in "
              f"save_roles={save_roles} at the given context length.\n"
              f"Consider increasing context_len or adjusting save_roles if this is unexpected.")
    return filtered
