import hashlib
from dataclasses import dataclass
import json


_SHAREGPT_ROLE_MAP = {
    "human": "user",
    "gpt": "assistant",
    "system": "system",
    "tool": "tool",
}


def _detect_format(sample: dict) -> str:
    if "messages" in sample and isinstance(sample["messages"], list):
        if sample["messages"] and isinstance(sample["messages"][0], dict) and "role" in sample["messages"][0]:
            return "openai"
    if "conversations" in sample and isinstance(sample["conversations"], list):
        if sample["conversations"] and isinstance(sample["conversations"][0], dict) and "from" in sample["conversations"][0]:
            return "sharegpt"
    if "instruction" in sample and "output" in sample:
        return "alpaca"
    if "text" in sample and "messages" not in sample and "conversations" not in sample:
        return "completion"
    raise ValueError(
        f"Could not detect dataset format. Found keys: {set(sample.keys())}. "
        f"Supported formats: OpenAI messages, ShareGPT, Alpaca instruct, completion/raw text."
    )


def _normalize_sample(sample: dict, fmt: str) -> dict:
    if fmt == "openai":
        return sample

    if fmt == "sharegpt":
        messages = []
        if "init" in sample and sample["init"]:
            messages.append({"role": "system", "content": sample["init"]})
        for turn in sample["conversations"]:
            role = _SHAREGPT_ROLE_MAP.get(turn["from"], turn["from"])
            messages.append({"role": role, "content": turn["value"]})
        return {"messages": messages}

    if fmt == "alpaca":
        messages = []
        if sample.get("system"):
            messages.append({"role": "system", "content": sample["system"]})
        user_content = sample["instruction"]
        if sample.get("input"):
            user_content += "\n" + sample["input"]
        messages.append({"role": "user", "content": user_content})
        messages.append({"role": "assistant", "content": sample["output"]})
        return {"messages": messages}

    if fmt == "completion":
        return {"messages": [{"role": "assistant", "content": sample["text"]}]}

    raise ValueError(f"Unknown format: {fmt}")


def read_jsonl_lazy(file_path):
    fmt = None
    with open(file_path, 'r', encoding='utf-8') as f:
        line_number = 0
        for line in f:
            line_number += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_number}: {e.msg}. Line content: {line.strip()}")
                continue
            if fmt is None:
                fmt = _detect_format(data)
                if fmt != "openai":
                    print(f"Detected dataset format: {fmt}, normalizing to messages format")
            yield _normalize_sample(data, fmt)


def read_jsonl(file_path) -> list[dict]:
    data_list = []
    fmt = None
    with open(file_path, 'r', encoding='utf-8') as f:
        line_number = 0
        for line in f:
            line_number += 1
            try:
                data = json.loads(line)
            except json.JSONDecodeError as e:
                print(f"Error parsing JSON on line {line_number}: {e.msg}. Line content: {line.strip()}")
                continue
            if fmt is None:
                fmt = _detect_format(data)
                if fmt != "openai":
                    print(f"Detected dataset format: {fmt}, normalizing to messages format")
            data = _normalize_sample(data, fmt)
            if "id" not in data:
                data["id"] = len(data_list)
            data_list.append(data)
    return data_list


def compute_content_sha(messages: list[dict], save_roles: set[str]) -> str:
    """SHA256 of concatenated content text from saved roles."""
    parts = [msg["content"] for msg in messages if msg["role"] in save_roles and msg.get("content")]
    return hashlib.sha256("||".join(parts).encode("utf-8")).hexdigest()

@dataclass
class SyncResult:
    to_collect: list[int]
    to_delete: list[int]
    to_reindex: dict[int, int]


def sync_dataset(samples: list[dict], save_roles: set[str], hdf5_shas: dict[int, str]) -> SyncResult:
    jsonl_sha_to_id = {}
    jsonl_sha_to_idx = {}
    for idx, sample in enumerate(samples):
        sha = compute_content_sha(sample["messages"], save_roles)
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
