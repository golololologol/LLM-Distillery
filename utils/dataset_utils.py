import hashlib
import logging
import re
import uuid
from dataclasses import dataclass
import json

from classes.dataset_formats import normalize as normalize_sample, REGISTRY as DATASET_FORMATS


log = logging.getLogger(__name__)


def read_jsonl_lazy(file_path, dataset_format: str = "auto"):
    announced = False
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_number}: {e.msg}. Line content: {line.strip()}")
                continue
            normalized = normalize_sample(data, dataset_format)
            if not announced:
                print(f"  Loaded dataset (format={dataset_format})")
                announced = True
            yield normalized


def read_jsonl(file_path, dataset_format: str = "auto") -> list[dict]:
    data_list = []
    announced = False
    with open(file_path, 'r', encoding='utf-8') as f:
        for line_number, line in enumerate(f, start=1):
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_number}: {e.msg}. Line content: {line.strip()}")
                continue
            data = normalize_sample(data, dataset_format)
            if not announced:
                print(f"Loaded dataset (format={dataset_format})")
                announced = True
            if "id" not in data:
                data["id"] = len(data_list)
            data_list.append(data)
    return data_list


def compute_source_sha(
    messages: list[dict],
    save_roles: set[str],
) -> str:
    """SHA256 of concatenated content, reasoning, and tool_calls from saved roles."""
    parts = []
    for msg in messages:
        if msg["role"] not in save_roles:
            continue
        chunks = []
        if msg.get("content"):
            chunks.append(msg["content"])
        if msg.get("reasoning"):
            chunks.append(msg["reasoning"])
        if msg.get("tool_calls"):
            chunks.append(json.dumps(msg["tool_calls"], sort_keys=True))
        if chunks:
            parts.append("\x00".join(chunks))
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()

@dataclass
class SyncResult:
    to_collect: list[int]
    to_delete: list[int]
    to_reindex: dict[int, int]


def sync_dataset(
    samples: list[dict],
    save_roles: set[str],
    hdf5_shas: dict[int, str],
) -> SyncResult:
    jsonl_sha_to_id = {}
    jsonl_sha_to_idx = {}
    for idx, sample in enumerate(samples):
        sha = compute_source_sha(sample["messages"], save_roles)
        if sha not in jsonl_sha_to_id:
            jsonl_sha_to_id[sha] = sample["id"]
            jsonl_sha_to_idx[sha] = idx

    hdf5_sha_to_id = {sha: convo_id for convo_id, sha in hdf5_shas.items()}

    jsonl_shas = set(jsonl_sha_to_id)
    existing_shas = set(hdf5_sha_to_id)

    to_collect = [jsonl_sha_to_idx[sha] for sha in jsonl_shas - existing_shas]
    to_delete = [hdf5_sha_to_id[sha] for sha in existing_shas - jsonl_shas]

    to_reindex = {}
    for sha in jsonl_shas & existing_shas:
        old_id = hdf5_sha_to_id[sha]
        new_id = jsonl_sha_to_id[sha]
        if old_id != new_id:
            to_reindex[old_id] = new_id

    return SyncResult(to_collect=to_collect, to_delete=to_delete, to_reindex=to_reindex)


_INLINE_REASONING_PATTERNS = {
    "think_tags": [
        re.compile(r"<think>(.*?)</think>", re.DOTALL),
        re.compile(r"<\|begin_of_thought\|>(.*?)<\|end_of_thought\|>", re.DOTALL),
    ]
}


_VALID_ROLES = {"system", "user", "assistant", "tool"}


def _canonicalize_tool_calls(raw: list) -> list[dict]:
    out = []
    for tc in raw:
        if not isinstance(tc, dict):
            raise ValueError(f"tool_call must be a dict, got {type(tc).__name__}")

        if "function" in tc and isinstance(tc["function"], dict):
            fn = tc["function"]
            name = fn.get("name")
            arguments = fn.get("arguments")
            tc_id = tc.get("id")
        else:
            name = tc.get("name")
            arguments = tc.get("arguments")
            tc_id = tc.get("id")

        if not name or not isinstance(name, str):
            raise ValueError(f"tool_call missing or empty 'name': {tc}")

        if isinstance(arguments, dict):
            arguments_str = json.dumps(arguments, sort_keys=True, separators=(",", ":"))
        elif isinstance(arguments, str):
            # Re-serialize any JSON-parseable value (dict, list, number, bool, null, string)
            # so whitespace/key-order/escape differences don't split the merge group.
            try:
                parsed = json.loads(arguments)
                arguments_str = json.dumps(parsed, sort_keys=True, separators=(",", ":"))
            except (ValueError, TypeError):
                arguments_str = arguments
        elif arguments is None:
            arguments_str = "{}"
        else:
            raise ValueError(f"tool_call arguments must be dict or str, got {type(arguments).__name__}")

        if not tc_id:
            tc_id = f"call_{uuid.uuid4().hex[:12]}"

        out.append({"name": name, "arguments": arguments_str, "id": tc_id})
    return out


class DatasetCanonicalizer:
    def __init__(self, on_multi_think: str = "error", inline_reasoning_mode: str = "disabled"):
        if on_multi_think not in ("error", "first_only", "warn_first_only"):
            raise ValueError(
                f"on_multi_think must be 'error', 'first_only', or 'warn_first_only', got {on_multi_think!r}"
            )
        if inline_reasoning_mode not in ("disabled", "think_tags"):
            raise ValueError(
                f"inline_reasoning_mode must be 'disabled' or 'think_tags', got {inline_reasoning_mode!r}"
            )
        self.on_multi_think = on_multi_think
        self.inline_reasoning_mode = inline_reasoning_mode

    def canonicalize_messages(self, messages: list[dict], *, sample_id=None) -> list[dict]:
        result = []
        for i, msg in enumerate(messages):
            role = msg.get("role")
            if not isinstance(role, str):
                raise ValueError(f"message {i}: role must be a string, got {type(role).__name__}")
            if role not in _VALID_ROLES:
                raise ValueError(f"message {i}: role {role!r} not in {_VALID_ROLES}")

            content = msg.get("content")
            if content is not None and not isinstance(content, str):
                raise ValueError(f"message {i}: content must be str or None, got {type(content).__name__}")

            out = {
                "role": role,
                "content": content,
                "reasoning": None,
                "tool_calls": [],
                "tool_call_id": None,
            }

            if role == "tool":
                tci = msg.get("tool_call_id")
                if tci is not None and not isinstance(tci, str):
                    raise ValueError(f"message {i}: tool_call_id must be str or None")
                out["tool_call_id"] = tci

            if role == "assistant":
                reasoning = msg.get("reasoning")
                if reasoning is None:
                    reasoning = msg.get("reasoning_content")

                if reasoning is None and self.inline_reasoning_mode != "disabled" and isinstance(content, str) and content:
                    for pat in _INLINE_REASONING_PATTERNS[self.inline_reasoning_mode]:
                        matches = list(pat.finditer(content))
                        if not matches:
                            continue
                        if len(matches) >= 2:
                            if self.on_multi_think == "error":
                                raise ValueError(
                                    f"sample {sample_id} message {i}: {len(matches)} <think> blocks "
                                    f"in single assistant message (on_multi_think='error')"
                                )
                            if self.on_multi_think == "warn_first_only":
                                log.warning(
                                    f"sample {sample_id} message {i}: {len(matches)} <think> blocks; taking first"
                                )
                        first = matches[0]
                        reasoning = first.group(1)
                        content = content[:first.start()] + content[first.end():]
                        break

                out["reasoning"] = reasoning
                out["content"] = content

                raw_tcs = msg.get("tool_calls")
                if raw_tcs:
                    if not isinstance(raw_tcs, list):
                        raise ValueError(f"message {i}: tool_calls must be a list")
                    out["tool_calls"] = _canonicalize_tool_calls(raw_tcs)

                if not (out["content"] or out["reasoning"] or out["tool_calls"]):
                    raise ValueError(
                        f"sample {sample_id} message {i}: assistant message has no content, reasoning, or tool_calls"
                    )

                # Plan v8 §6: assistant messages with tool_calls and/or
                # reasoning but missing content materialise to ``content = ""``
                # so a (possibly zero-byte) content segment exists. Its post
                # anchor captures the "skip content, jump to next" decision
                # that the byte channel cannot represent on its own.
                if out["content"] is None:
                    out["content"] = ""

            result.append(out)
        return result

    def canonicalize_sample(self, sample: dict) -> dict:
        return {**sample, "messages": self.canonicalize_messages(sample["messages"], sample_id=sample.get("id"))}
