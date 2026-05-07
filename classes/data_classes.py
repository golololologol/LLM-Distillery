from dataclasses import dataclass
from multiprocessing import shared_memory
from typing import Any
import numpy as np


@dataclass
class Segment:
    msg_idx: int
    type: str
    byte_start: int
    byte_end: int
    canonical_target_hash: str = ""
    truncated: bool = False
    # v8 extensions (plan §6). Defaults preserve v7 semantics when fields are
    # absent in legacy manifests / synthetic constructions.
    segment_idx: int = 0
    tool_call_idx: int | None = None
    role: str = "assistant"
    handling: str = "active"
    pre_event_row: int | None = None
    post_event_row: int | None = None
    byte_low_confidence: bool = False

    @property
    def byte_count(self) -> int:
        return self.byte_end - self.byte_start

    def to_manifest_entry(self) -> dict:
        entry = {
            "msg_idx": self.msg_idx,
            "type": self.type,
            "byte_count": self.byte_count,
            "canonical_target_hash": self.canonical_target_hash,
            "truncated": self.truncated,
        }
        # v8 fields are only emitted when they carry non-default information,
        # keeping the manifest backward-compatible with v7 readers/tests.
        if self.segment_idx:
            entry["segment_idx"] = self.segment_idx
        if self.tool_call_idx is not None:
            entry["tool_call_idx"] = self.tool_call_idx
        if self.role and self.role != "assistant":
            entry["role"] = self.role
        if self.handling and self.handling != "active":
            entry["handling"] = self.handling
        if self.pre_event_row is not None:
            entry["pre_event_row"] = self.pre_event_row
        if self.post_event_row is not None:
            entry["post_event_row"] = self.post_event_row
        if self.byte_low_confidence:
            entry["byte_low_confidence"] = True
        return entry

    @classmethod
    def from_manifest(cls, manifest: list[dict]) -> list["Segment"]:
        segments = []
        offset = 0
        for entry in manifest:
            bc = entry["byte_count"]
            segments.append(cls(
                msg_idx=entry["msg_idx"],
                type=entry["type"],
                byte_start=offset,
                byte_end=offset + bc,
                canonical_target_hash=entry["canonical_target_hash"],
                truncated=entry["truncated"],
                segment_idx=entry.get("segment_idx", 0),
                tool_call_idx=entry.get("tool_call_idx"),
                role=entry.get("role", "assistant"),
                handling=entry.get("handling", "active"),
                pre_event_row=entry.get("pre_event_row"),
                post_event_row=entry.get("post_event_row"),
                byte_low_confidence=bool(entry.get("byte_low_confidence", False)),
            ))
            offset += bc
        return segments


@dataclass
class Anchor:
    """An event-channel anchor point bound to a specific segment (plan v8 §5).

    ``anchor_side`` is ``"pre"`` (token position before the segment's first
    byte, captures the decision to *enter* the segment) or ``"post"`` (token
    position after the last byte, captures the decision about what comes
    next). ``token_pos`` is the absolute index into the rendered token
    stream; ``unreachable`` is True when the position falls outside
    ``[0, length)`` (typically due to cropping).
    """
    msg_idx: int
    segment_idx: int
    type: str
    tool_call_idx: int | None
    anchor_side: str            # "pre" | "post"
    token_pos: int
    unreachable: bool = False
    truncated: bool = False

    def to_manifest_entry(self) -> dict:
        d = {
            "msg_idx": self.msg_idx,
            "segment_idx": self.segment_idx,
            "anchor_side": self.anchor_side,
        }
        if self.type:
            d["type"] = self.type
        if self.tool_call_idx is not None:
            d["tool_call_idx"] = self.tool_call_idx
        if self.unreachable:
            d["unreachable"] = True
        if self.truncated:
            d["truncated"] = True
        return d


@dataclass
class ConvoProcessed:
    origin_convo_id: int | str
    tokens: np.ndarray
    segments: list[Segment]
    content_sha: str
    formatted_text: str
    padding: int
    cropped: bool
    length: int
    anchors: list[Anchor] = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.anchors is None:
            self.anchors = []

    def byte_ranges(self) -> list[tuple[int, int]]:
        return [(s.byte_start, s.byte_end) for s in self.segments]

    def actual_bytes(self) -> np.ndarray:
        text = self.formatted_text.encode("utf-8")
        parts = [text[s.byte_start:s.byte_end] for s in self.segments]
        if not parts:
            return np.array([], dtype=np.uint8)
        return np.frombuffer(b"".join(parts), dtype=np.uint8)

    def filter_by_handling(self, handling: dict[str, str], mode: str) -> list[Segment]:
        return [s for s in self.segments if handling.get(s.type) == mode]

    def usable_anchors(self, length: int | None = None) -> list[tuple[int, dict]]:
        """Return ``[(token_pos, manifest_entry), ...]`` for anchors that fall
        within the usable token range. Used by inference workers to build the
        ``anchor_token_positions`` / ``event_manifest_template`` arrays.
        """
        L = self.length if length is None else length
        out: list[tuple[int, dict]] = []
        for a in self.anchors or ():
            tp = int(a.token_pos)
            unreachable = bool(a.unreachable) or tp < 0 or tp >= L
            entry = a.to_manifest_entry()
            if unreachable:
                entry["unreachable"] = True
                continue  # skip entirely; not usable
            out.append((tp, entry))
        return out


class Distribution:
    def __init__(
        self,
        origin_convo_id: int | str,
        content_sha: str = "",
        cropped: bool = False,
        distribution: np.ndarray | None = None,
    ):
        self.origin_convo_id = origin_convo_id
        self.content_sha = content_sha
        self.cropped = cropped
        self.distribution = distribution
        self.segment_manifest: list[dict] | None = None
        self.shd_mem_name: str = ""
        self.distr_shape: tuple[int, ...] | None = None
        self.distr_dtype: np.dtype[Any] | None = None

        # v8 event-channel payload (plan §7).  ``events`` is a ``[N, |E|]`` fp16
        # array stored in its own shared-memory segment; ``event_manifest`` is
        # a list of dicts (one per anchor) that mirrors what the merger / data
        # manager need to write into the HDF5 ``event_manifest`` attr.
        self.events: np.ndarray | None = None
        self.events_shm_name: str = ""
        self.events_shape: tuple[int, ...] | None = None
        self.events_dtype: np.dtype[Any] | None = None
        self.event_manifest: list[dict] | None = None
        self.event_alphabet: tuple[str, ...] | None = None
        self.supported_mask: int = 0
        self.specials_hash: str | None = None

    def to_shd_mem(self) -> shared_memory.SharedMemory:
        """
        Moves the distribution to shared memory and updates the metadata accordingly.
        Incurs one copy from the original numpy array to shared memory.
        """
        if self.distribution is None:
            raise RuntimeError("Distribution.to_shd_mem called without distribution data")
        distribution = self.distribution
        shd_mem = shared_memory.SharedMemory(create=True, size=distribution.nbytes)
        self.shd_mem_name = shd_mem.name
        self.distr_shape = tuple(distribution.shape)
        self.distr_dtype = distribution.dtype
        shd_distr = np.ndarray(self.distr_shape, dtype=self.distr_dtype, buffer=shd_mem.buf)
        shd_distr[:] = distribution
        self.distribution = None
        return shd_mem

    def to_shd_mem_gpu(self, tensor) -> shared_memory.SharedMemory:
        """
        Moves a PyTorch tensor to shared memory and updates the distribution metadata accordingly.
        Incurrs one copy: GPU -> CPU
        """
        import torch
        _TORCH_TO_NP = {torch.float16: np.dtype(np.float16), torch.float32: np.dtype(np.float32), torch.float64: np.dtype(np.float64)}
        self.distr_shape = tuple(int(dim) for dim in tensor.shape)
        self.distr_dtype = _TORCH_TO_NP[tensor.dtype]
        shm = shared_memory.SharedMemory(
            create=True, size=tensor.nelement() * tensor.element_size()
        )
        torch.from_numpy(np.ndarray(self.distr_shape, dtype=self.distr_dtype, buffer=shm.buf)).copy_(tensor)
        self.shd_mem_name = shm.name
        self.distribution = None
        return shm

    def from_shd_mem(self) -> shared_memory.SharedMemory:
        """
        Maps the shared memory back to a numpy array and assigns it to self.distribution.
        Incurrs zero copies.
        """
        if self.distr_shape is None or self.distr_dtype is None:
            raise RuntimeError("Distribution.from_shd_mem called before shared-memory metadata was set")
        shd_mem = shared_memory.SharedMemory(name=self.shd_mem_name)
        self.distribution = np.ndarray(self.distr_shape, dtype=self.distr_dtype, buffer=shd_mem.buf)
        return shd_mem

    # --- v8 event-channel shared-memory helpers --------------------------

    def events_to_shd_mem_gpu(self, tensor) -> shared_memory.SharedMemory | None:
        """Move a GPU event tensor into shared memory.

        Returns the SharedMemory segment so the caller can keep it alive until
        the data manager has consumed it. Returns ``None`` for empty tensors.
        """
        import torch
        _TORCH_TO_NP = {torch.float16: np.dtype(np.float16), torch.float32: np.dtype(np.float32), torch.float64: np.dtype(np.float64)}
        if tensor is None or tensor.numel() == 0:
            self.events_shape = (0, 0)
            self.events_dtype = np.dtype(np.float16)
            self.events_shm_name = ""
            return None
        self.events_shape = tuple(tensor.shape)
        self.events_dtype = _TORCH_TO_NP[tensor.dtype]
        shm = shared_memory.SharedMemory(
            create=True, size=tensor.nelement() * tensor.element_size(),
        )
        torch.from_numpy(np.ndarray(self.events_shape, dtype=self.events_dtype, buffer=shm.buf)).copy_(tensor)
        self.events_shm_name = shm.name
        self.events = None
        return shm

    def events_to_shd_mem(self) -> shared_memory.SharedMemory | None:
        """Move an in-memory ``self.events`` numpy array into shared memory."""
        if self.events is None or self.events.size == 0:
            self.events_shape = (0, 0)
            self.events_dtype = np.dtype(np.float16)
            self.events_shm_name = ""
            return None
        shm = shared_memory.SharedMemory(create=True, size=self.events.nbytes)
        self.events_shm_name = shm.name
        self.events_shape = tuple(self.events.shape)
        self.events_dtype = self.events.dtype
        view = np.ndarray(self.events_shape, dtype=self.events_dtype, buffer=shm.buf)
        view[:] = self.events
        self.events = None
        return shm

    def events_from_shd_mem(self) -> shared_memory.SharedMemory | None:
        """Map ``events`` back to a numpy array (zero-copy)."""
        if not self.events_shm_name:
            self.events = None
            return None
        if self.events_shape is None or self.events_dtype is None:
            raise RuntimeError("Distribution.events_from_shd_mem called before shared-memory metadata was set")
        shm = shared_memory.SharedMemory(name=self.events_shm_name)
        self.events = np.ndarray(self.events_shape, dtype=self.events_dtype, buffer=shm.buf)
        return shm


class BatchResult:
    
    def __init__(self, shm_name, shape, dtype, padding):
        self._shm = shared_memory.SharedMemory(name=shm_name)
        self.data = np.ndarray(shape, dtype=dtype, buffer=self._shm.buf)
        self.padding = padding

    def close(self):
        self._shm.close()
        self._shm.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@dataclass
class TrainingState:
    num_trained: int
    epoch: int
    next_accum: int
    next_val: int
    next_save: int
    next_state_save: int | None
    wandb_run_id: str | None = None
    best_val_loss: float | None = None
