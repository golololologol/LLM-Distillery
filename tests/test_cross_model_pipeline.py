import json

import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pytest

from classes.formatter import SentinelFormatter
from classes.preprocessing import preprocess_samples
from utils.dataset_utils import DatasetCanonicalizer
from classes.dataset_formats import normalize as _normalize_sample
from utils.merging_utils import _merge_one_convo


OPENAI_SAMPLE = {"messages": [
    {"role": "user", "content": "Hello"},
    {"role": "assistant", "content": "Hi there!"},
]}

SHAREGPT_SAMPLE = {"conversations": [
    {"from": "human", "value": "Hello"},
    {"from": "gpt", "value": "Hi there!"},
]}


def test_end_to_end_pipeline(tokenizer, tmp_path):
    fmt = SentinelFormatter(tokenizer)

    sample = _normalize_sample(OPENAI_SAMPLE, "auto")
    sample["id"] = 0
    canon = DatasetCanonicalizer().canonicalize_sample(sample)
    assert canon["messages"][1]["content"] == "Hi there!"

    processed = preprocess_samples([canon], fmt, context_len=256)
    assert len(processed) == 1
    r = processed[0]
    content_segs = [s for s in r.segments if s.type == "content"]
    assert content_segs
    expected_hash = content_segs[0].canonical_target_hash
    assert len(expected_hash) == 64

    with h5py.File(tmp_path / "A.hdf5", "w") as f:
        f.attrs["layout_version"] = 4
        g = f.create_group("convo_0")
        bc = content_segs[0].byte_count
        data = np.random.dirichlet(np.ones(256), size=bc).astype(np.float16)
        g.create_dataset("dense_distributions", data=data)
        g.attrs["content_sha"] = r.content_sha
        g.attrs["segment_manifest"] = json.dumps([{
            "msg_idx": content_segs[0].msg_idx,
            "type": "content",
            "byte_count": bc,
            "canonical_target_hash": expected_hash,
            "truncated": False,
        }])

    merged = _merge_one_convo(str(tmp_path), {"A": 1.0}, "convo_0")
    assert merged is not None
    _, compressed, sha, manifest_json, shape, _rejects, _seg_count, _ev, _evm, _ea, _um = merged
    assert sha == r.content_sha
    mm = json.loads(manifest_json)
    assert mm[0]["canonical_target_hash"] == expected_hash

    teacher_lookup = {(e["msg_idx"], e["type"]): e for e in mm}
    seg = content_segs[0]
    te = teacher_lookup[(seg.msg_idx, seg.type)]
    assert te["canonical_target_hash"] == seg.canonical_target_hash


def test_sharegpt_canonicalizes_and_processes(tokenizer):
    fmt = SentinelFormatter(tokenizer)
    sample = _normalize_sample(SHAREGPT_SAMPLE, "auto")
    sample["id"] = 0
    canon = DatasetCanonicalizer().canonicalize_sample(sample)
    out = preprocess_samples([canon], fmt, context_len=256)
    assert len(out) == 1
    assert any(s.type == "content" for s in out[0].segments)


def test_pipeline_preserves_segment_bytes(tokenizer):
    fmt = SentinelFormatter(tokenizer)
    sample = {"id": 0, "messages": [
        {"role": "user", "content": "x"},
        {"role": "assistant", "content": "specific-marker-text"},
    ]}
    canon = DatasetCanonicalizer().canonicalize_sample(sample)
    out = preprocess_samples([canon], fmt, context_len=256)
    r = out[0]
    full = r.formatted_text.encode("utf-8")
    content_segs = [s for s in r.segments if s.type == "content"]
    assert any(full[s.byte_start:s.byte_end].decode("utf-8") == "specific-marker-text" for s in content_segs)
