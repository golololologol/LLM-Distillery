import json
import os

import h5py
import hdf5plugin  # noqa: F401 - registers zstd filter
import numpy as np
import pytest
import zstandard
from typing import Any, cast

from utils.merging_utils import _merge_one_convo, merge_teacher_files


def _write_teacher(path, segs_by_convo, *, seed=0):
    """segs_by_convo: {convo_id: [(msg_idx, seg_type, byte_count, hash, truncated, fill)]}"""
    rng = np.random.default_rng(seed)
    with h5py.File(path, "w") as f:
        f.attrs["layout_version"] = 5
        for cid, segs in segs_by_convo.items():
            g = f.create_group(f"convo_{cid}")
            rows = []
            manifest = []
            for seg_idx, (msg_idx, seg_type, bc, h, trunc, fill) in enumerate(segs):
                arr = np.full((bc, 256), fill, dtype=np.float32)
                arr = arr + rng.standard_normal((bc, 256)).astype(np.float32) * 0.001
                arr = np.abs(arr)
                arr /= arr.sum(axis=1, keepdims=True)
                rows.append(arr.astype(np.float16))
                manifest.append({
                    "msg_idx": msg_idx,
                    "segment_idx": seg_idx,
                    "type": seg_type,
                    "tool_call_idx": None,
                    "byte_count": bc,
                    "canonical_target_hash": h,
                    "truncated": trunc,
                    "handling": "active",
                    "role": "assistant",
                })
            data = np.concatenate(rows, axis=0) if rows else np.zeros((0, 256), dtype=np.float16)
            g.create_dataset("dense_distributions", data=data)
            g.attrs["content_sha"] = f"sha_{cid}"
            g.attrs["segment_manifest"] = json.dumps(manifest)


def _run_merge(tmp_path, weights):
    result = _merge_one_convo(str(tmp_path), weights, "convo_0")
    assert result is not None
    _, compressed, _, manifest_json, shape, _rejects, _seg_count, _ev, _evm, _ea, _um = result
    raw = zstandard.ZstdDecompressor().decompress(compressed)
    dist = np.frombuffer(raw, dtype=np.float16).reshape(shape)
    return dist, json.loads(manifest_json)


def test_hash_match_weighted_average_full_length(tmp_path):
    _write_teacher(tmp_path / "A.hdf5", {0: [(0, "content", 10, "hA", False, 0.5)]}, seed=1)
    _write_teacher(tmp_path / "B.hdf5", {0: [(0, "content", 10, "hA", False, 0.5)]}, seed=2)

    dist, manifest = _run_merge(tmp_path, {"A": 0.5, "B": 0.5})
    assert manifest[0]["msg_idx"] == 0
    assert manifest[0]["type"] == "content"
    assert manifest[0]["byte_count"] == 10
    assert manifest[0]["canonical_target_hash"] == "hA"
    assert manifest[0]["truncated"] is False
    assert dist.shape == (10, 256)


def test_hash_match_all_truncated_marked_truncated(tmp_path):
    _write_teacher(tmp_path / "A.hdf5", {0: [(0, "content", 10, "hA", True, 0.5)]}, seed=1)
    _write_teacher(tmp_path / "B.hdf5", {0: [(0, "content", 10, "hA", True, 0.5)]}, seed=2)

    dist, manifest = _run_merge(tmp_path, {"A": 0.5, "B": 0.5})
    assert manifest[0]["truncated"] is True
    assert manifest[0]["byte_count"] == 10


def test_hash_mismatch_minority_rejected(tmp_path):
    _write_teacher(tmp_path / "A.hdf5", {0: [(0, "content", 10, "hA", False, 0.5)]}, seed=1)
    _write_teacher(tmp_path / "B.hdf5", {0: [(0, "content", 10, "hA", False, 0.5)]}, seed=2)
    _write_teacher(tmp_path / "C.hdf5", {0: [(0, "content", 10, "hBAD", False, 0.5)]}, seed=3)

    dist, manifest = _run_merge(tmp_path, {"A": 0.4, "B": 0.4, "C": 0.4})
    assert manifest[0]["canonical_target_hash"] == "hA"


def test_hash_tie_highest_summed_weight_wins(tmp_path):
    # 2 teachers hash A (weights 0.1+0.1), 2 teachers hash B (weights 0.5+0.5). Same group size.
    _write_teacher(tmp_path / "A.hdf5", {0: [(0, "content", 10, "hA", False, 0.5)]}, seed=1)
    _write_teacher(tmp_path / "B.hdf5", {0: [(0, "content", 10, "hA", False, 0.5)]}, seed=2)
    _write_teacher(tmp_path / "C.hdf5", {0: [(0, "content", 10, "hB", False, 0.5)]}, seed=3)
    _write_teacher(tmp_path / "D.hdf5", {0: [(0, "content", 10, "hB", False, 0.5)]}, seed=4)

    dist, manifest = _run_merge(tmp_path, {"A": 0.1, "B": 0.1, "C": 0.5, "D": 0.5})
    assert manifest[0]["canonical_target_hash"] == "hB"


def test_hash_weight_tie_lex_earliest_teacher_wins(tmp_path):
    # Both groups: same size (2), same summed weight. Tie → lex-earliest teacher name.
    _write_teacher(tmp_path / "A.hdf5", {0: [(0, "content", 10, "hA", False, 0.5)]}, seed=1)
    _write_teacher(tmp_path / "D.hdf5", {0: [(0, "content", 10, "hA", False, 0.5)]}, seed=2)
    _write_teacher(tmp_path / "B.hdf5", {0: [(0, "content", 10, "hB", False, 0.5)]}, seed=3)
    _write_teacher(tmp_path / "C.hdf5", {0: [(0, "content", 10, "hB", False, 0.5)]}, seed=4)

    dist, manifest = _run_merge(tmp_path, {"A": 0.3, "D": 0.3, "B": 0.3, "C": 0.3})
    # Group hA min name = "A", group hB min name = "B" → A wins
    assert manifest[0]["canonical_target_hash"] == "hA"


def test_truncated_prefix_merge_with_full_length(tmp_path):
    # A full-length non-truncated 10; B truncated 6 → merged over full 10, B contributes 0..6.
    _write_teacher(tmp_path / "A.hdf5", {0: [(0, "content", 10, "hA", False, 0.5)]}, seed=1)
    _write_teacher(tmp_path / "B.hdf5", {0: [(0, "content", 6, "hA", True, 0.5)]}, seed=2)

    dist, manifest = _run_merge(tmp_path, {"A": 0.5, "B": 0.5})
    assert manifest[0]["byte_count"] == 10
    assert manifest[0]["truncated"] is False


def test_non_truncated_short_rejected(tmp_path):
    # A full-length 10 non-truncated; B short 6 non-truncated (real disagreement).
    _write_teacher(tmp_path / "A.hdf5", {0: [(0, "content", 10, "hA", False, 0.5)]}, seed=1)
    _write_teacher(tmp_path / "B.hdf5", {0: [(0, "content", 6, "hA", False, 0.5)]}, seed=2)

    dist, manifest = _run_merge(tmp_path, {"A": 0.5, "B": 0.5})
    assert manifest[0]["byte_count"] == 10
    # B's contribution is rejected → only A present; final has no B-contribution region
    assert dist.shape == (10, 256)


def test_tool_call_strict_hash_mismatch_excludes(tmp_path):
    _write_teacher(tmp_path / "A.hdf5", {0: [(0, "tool_call", 10, "hA", False, 0.5)]}, seed=1)
    _write_teacher(tmp_path / "B.hdf5", {0: [(0, "tool_call", 10, "hA", False, 0.5)]}, seed=2)
    _write_teacher(tmp_path / "C.hdf5", {0: [(0, "tool_call", 10, "hOTHER", False, 0.5)]}, seed=3)

    dist, manifest = _run_merge(tmp_path, {"A": 0.5, "B": 0.5, "C": 0.5})
    assert manifest[0]["canonical_target_hash"] == "hA"


def test_tool_call_never_prefix_merge(tmp_path):
    # Tool_call: hashes match but lengths differ (A=10, B=8 truncated). Tool_call should not prefix-merge.
    _write_teacher(tmp_path / "A.hdf5", {0: [(0, "tool_call", 10, "hA", False, 0.5)]}, seed=1)
    _write_teacher(tmp_path / "B.hdf5", {0: [(0, "tool_call", 8, "hA", True, 0.5)]}, seed=2)

    dist, manifest = _run_merge(tmp_path, {"A": 0.5, "B": 0.5})
    # Strict length: B (shorter) is dropped as tool_call_length_mismatch, A kept.
    assert manifest[0]["byte_count"] == 10


def test_sparse_contributor_set(tmp_path):
    # Segment missing from one teacher - no error.
    _write_teacher(tmp_path / "A.hdf5", {0: [(0, "content", 10, "hA", False, 0.5),
                                              (1, "content", 5, "hB", False, 0.5)]}, seed=1)
    _write_teacher(tmp_path / "B.hdf5", {0: [(0, "content", 10, "hA", False, 0.5)]}, seed=2)

    dist, manifest = _run_merge(tmp_path, {"A": 0.5, "B": 0.5})
    keys = [(m["msg_idx"], m["type"]) for m in manifest]
    assert (0, "content") in keys
    assert (1, "content") in keys


def test_layout_version_mismatch_fails(tmp_path):
    # File without layout_version attribute.
    with h5py.File(tmp_path / "A.hdf5", "w") as f:
        g = f.create_group("convo_0")
        g.create_dataset("dense_distributions", data=np.zeros((1, 256), dtype=np.float16))
        g.attrs["content_sha"] = "x"
        g.attrs["segment_manifest"] = json.dumps([])
    with pytest.raises(RuntimeError, match="expected layout_version>=4"):
        _merge_one_convo(str(tmp_path), {"A": 1.0}, "convo_0")


def test_merge_teacher_files_end_to_end(tmp_path):
    _write_teacher(tmp_path / "A.hdf5", {0: [(0, "content", 10, "hA", False, 0.5)]}, seed=1)
    _write_teacher(tmp_path / "B.hdf5", {0: [(0, "content", 10, "hA", False, 0.5)]}, seed=2)
    merge_teacher_files(str(tmp_path), {"A": 0.5, "B": 0.5}, label="test")
    assert os.path.exists(tmp_path / "_merged.hdf5")
    with h5py.File(tmp_path / "_merged.hdf5", "r") as f:
        assert f.attrs["layout_version"] == 5
        assert "convo_0" in f
        g = cast(h5py.Group, f["convo_0"])
        m = json.loads(cast(Any, g.attrs["segment_manifest"]))
        assert m[0]["canonical_target_hash"] == "hA"


def test_reject_counts_returned_from_worker(tmp_path):
    # 2 teachers agree on hA, 1 teacher disagrees with hBAD → 1 hash_mismatch reject.
    _write_teacher(tmp_path / "A.hdf5", {0: [(0, "content", 10, "hA", False, 0.5)]}, seed=1)
    _write_teacher(tmp_path / "B.hdf5", {0: [(0, "content", 10, "hA", False, 0.5)]}, seed=2)
    _write_teacher(tmp_path / "C.hdf5", {0: [(0, "content", 10, "hBAD", False, 0.5)]}, seed=3)
    result = _merge_one_convo(str(tmp_path), {"A": 0.4, "B": 0.4, "C": 0.4}, "convo_0")
    assert result is not None
    rejects = result[5]
    seg_count = result[6]
    assert rejects.get("hash_mismatch") == 1
    assert seg_count == 3


def test_reject_counts_tool_call_length_mismatch(tmp_path):
    _write_teacher(tmp_path / "A.hdf5", {0: [(0, "tool_call", 10, "hA", False, 0.5)]}, seed=1)
    _write_teacher(tmp_path / "B.hdf5", {0: [(0, "tool_call", 8, "hA", True, 0.5)]}, seed=2)
    result = _merge_one_convo(str(tmp_path), {"A": 0.5, "B": 0.5}, "convo_0")
    assert result is not None
    rejects = result[5]
    assert rejects.get("tool_call_length_mismatch") == 1


def test_reject_counts_non_truncated_short(tmp_path):
    _write_teacher(tmp_path / "A.hdf5", {0: [(0, "content", 10, "hA", False, 0.5)]}, seed=1)
    _write_teacher(tmp_path / "B.hdf5", {0: [(0, "content", 6, "hA", False, 0.5)]}, seed=2)
    result = _merge_one_convo(str(tmp_path), {"A": 0.5, "B": 0.5}, "convo_0")
    assert result is not None
    rejects = result[5]
    assert rejects.get("non_truncated_short") == 1


def test_high_reject_rate_warning_logged(tmp_path, caplog):
    import logging as _logging
    # Build 1 agreeing + 1 disagreeing → 1 reject out of 2 contributions = 50%.
    _write_teacher(tmp_path / "A.hdf5", {0: [(0, "content", 10, "hA", False, 0.5)]}, seed=1)
    _write_teacher(tmp_path / "B.hdf5", {0: [(0, "content", 10, "hBAD", False, 0.5)]}, seed=2)
    with caplog.at_level(_logging.WARNING, logger="utils.merging_utils"):
        merge_teacher_files(str(tmp_path), {"A": 0.5, "B": 0.5}, label="test_high_reject")
    warnings = [r for r in caplog.records if r.levelno >= _logging.WARNING]
    assert any("merge reject" in r.getMessage().lower() for r in warnings)
