"""Plan v8 §7: H5DataManager._save_data must persist event channel + new attrs."""
import json
import h5py
import hdf5plugin  # noqa: F401
import numpy as np
import pytest
from typing import Any, cast

from classes.data_manager import H5DataManager


@pytest.fixture
def dm(tmp_path):
    """Construct a manager pointed at a fresh tmp HDF5 file (no spawned process)."""
    path = tmp_path / "events.hdf5"
    # Write directly through ``_save_data`` rather than the multiprocess
    # loading process, since we just want to exercise the save path.
    mgr = H5DataManager.__new__(H5DataManager)  # type: ignore[call-arg]
    mgr.file_path = str(path)
    mgr.read_only = False
    # Attributes touched by H5DataManager.close()/__del__ when the spawned
    # loading process is bypassed.
    mgr.queue = None
    mgr.result_queue = None
    mgr.loading_process = None
    mgr.shared_batches = []
    return mgr


def test_save_writes_event_channel_and_attrs(dm, tmp_path):
    data = np.full((3, 256), 1.0 / 256, dtype=np.float16)
    events = np.array([[0.7, 0.3], [0.5, 0.5]], dtype=np.float16)
    manifest = [
        {"msg_idx": 0, "segment_idx": 0, "anchor_side": "pre", "row": 0},
        {"msg_idx": 0, "segment_idx": 0, "anchor_side": "post", "row": 1},
    ]
    seg_manifest = [{
        "msg_idx": 0, "type": "content", "byte_count": 3,
        "canonical_target_hash": "h", "truncated": False,
        "pre_event_row": 0, "post_event_row": 1,
    }]
    alphabet = ("E_NONE", "E_END_TURN")

    with h5py.File(dm.file_path, "a") as f:
        dm._save_data(
            f, data, convo_id=0, content_sha="sha", cropped=False,
            segment_manifest=seg_manifest,
            events=events, event_manifest=manifest,
            event_alphabet=alphabet, supported_mask=0b11,
            specials_hash="hash_v",
        )

    with h5py.File(dm.file_path, "r") as f:
        assert int(cast(Any, f.attrs["layout_version"])) == 6
        assert f.attrs["specials_hash"] == "hash_v"
        # event_alphabet is stored as an ndarray of strings (one per slot).
        ea = cast(Any, f.attrs["event_alphabet"])
        assert [str(x) for x in ea] == list(alphabet)
        g = cast(h5py.Group, f["convo_0"])
        assert "event_distributions" in g
        event_ds = cast(h5py.Dataset, g["event_distributions"])
        np.testing.assert_allclose(event_ds[:].astype(np.float32),
                                   events.astype(np.float32), atol=1e-4)
        assert json.loads(cast(Any, g.attrs["event_manifest"])) == manifest
        assert int(cast(Any, g.attrs["event_count"])) == 2
        assert int(cast(Any, g.attrs["supported_mask"])) == 0b11
        assert g.attrs["specials_hash"] == "hash_v"


def test_save_without_events_keeps_layout_v5(dm):
    data = np.full((1, 256), 1.0 / 256, dtype=np.float16)
    with h5py.File(dm.file_path, "a") as f:
        dm._save_data(f, data, convo_id=1, content_sha="sha2", cropped=False)
    with h5py.File(dm.file_path, "r") as f:
        # No layout bump when nothing v8-specific is written.
        assert "layout_version" not in f.attrs or int(cast(Any, f.attrs["layout_version"])) < 6
        g = cast(h5py.Group, f["convo_1"])
        assert int(g.attrs.get("event_count", 0)) == 0
        assert "event_distributions" not in g
