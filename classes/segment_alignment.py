from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

from classes.data_classes import Anchor, ConvoProcessed, Segment


@dataclass(frozen=True, order=True)
class SegmentKey:
    msg_idx: int
    segment_idx: int
    type: str
    tool_call_idx: int | None = None

    @classmethod
    def from_segment(cls, segment: Segment) -> "SegmentKey":
        return cls(
            msg_idx=int(segment.msg_idx),
            segment_idx=int(segment.segment_idx),
            type=str(segment.type),
            tool_call_idx=segment.tool_call_idx,
        )

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any]) -> "SegmentKey":
        return cls(
            msg_idx=int(entry["msg_idx"]),
            segment_idx=int(entry.get("segment_idx", 0)),
            type=str(entry["type"]),
            tool_call_idx=entry.get("tool_call_idx"),
        )

    def legacy_fallback(self) -> "SegmentKey":
        return SegmentKey(self.msg_idx, 0, self.type, None)


@dataclass(frozen=True, order=True)
class AnchorKey:
    msg_idx: int
    segment_idx: int
    anchor_side: str

    @classmethod
    def from_anchor(cls, anchor: Anchor) -> "AnchorKey":
        return cls(
            msg_idx=int(anchor.msg_idx),
            segment_idx=int(anchor.segment_idx),
            anchor_side=str(anchor.anchor_side),
        )

    @classmethod
    def from_entry(cls, entry: Mapping[str, Any]) -> "AnchorKey":
        return cls(
            msg_idx=int(entry["msg_idx"]),
            segment_idx=int(entry.get("segment_idx", -1)),
            anchor_side=str(entry["anchor_side"]),
        )


@dataclass(frozen=True)
class ByteAlignment:
    train_ranges: list[tuple[int, int]]
    teacher_slices: list[tuple[int, int]]

    def actual_bytes(self, formatted_text: str) -> np.ndarray:
        text_bytes = formatted_text.encode("utf-8")
        parts = [text_bytes[start:end] for start, end in self.train_ranges]
        if not parts:
            return np.array([], dtype=np.uint8)
        return np.frombuffer(b"".join(parts), dtype=np.uint8)


@dataclass(frozen=True)
class EventAlignment:
    positions: tuple[int, ...]
    teacher_rows: tuple[int, ...]


def build_segment_lookup(manifest: Sequence[Mapping[str, Any]]) -> dict[SegmentKey, tuple[int, int, str, bool]]:
    lookup: dict[SegmentKey, tuple[int, int, str, bool]] = {}
    row = 0
    for entry in manifest:
        byte_count = int(entry["byte_count"])
        lookup[SegmentKey.from_entry(entry)] = (
            row,
            row + byte_count,
            str(entry.get("canonical_target_hash", "")),
            bool(entry.get("truncated", False)),
        )
        row += byte_count
    return lookup


def align_byte_segments(
    convo: ConvoProcessed,
    manifest: Sequence[Mapping[str, Any]],
    segment_handling: Mapping[str, str],
) -> ByteAlignment:
    lookup = build_segment_lookup(manifest)
    train_ranges: list[tuple[int, int]] = []
    teacher_slices: list[tuple[int, int]] = []

    for segment in convo.segments:
        if segment_handling.get(segment.type) != "active":
            continue

        key = SegmentKey.from_segment(segment)
        teacher_segment = lookup.get(key) or lookup.get(key.legacy_fallback())
        if teacher_segment is None:
            continue
        teacher_start, teacher_end, teacher_hash, teacher_truncated = teacher_segment

        student_hash = segment.canonical_target_hash
        if student_hash and teacher_hash and student_hash != teacher_hash:
            raise RuntimeError(
                f"canonical_target_hash mismatch: convo={convo.origin_convo_id} "
                f"msg_idx={segment.msg_idx} segment_idx={segment.segment_idx} type={segment.type} "
                f"student={student_hash[:12]} teacher={teacher_hash[:12]}"
            )

        student_bytes = int(segment.byte_count)
        teacher_bytes = teacher_end - teacher_start
        byte_count = min(student_bytes, teacher_bytes)
        if not segment.truncated and not teacher_truncated and student_bytes != teacher_bytes:
            raise RuntimeError(
                f"full-length byte_count mismatch: convo={convo.origin_convo_id} "
                f"msg_idx={segment.msg_idx} type={segment.type} "
                f"student={student_bytes} teacher={teacher_bytes}"
            )
        if byte_count > 0:
            train_ranges.append((segment.byte_start, segment.byte_start + byte_count))
            teacher_slices.append((teacher_start, teacher_start + byte_count))

    return ByteAlignment(train_ranges, teacher_slices)


def align_event_anchors(
    convo: ConvoProcessed,
    event_manifest: Sequence[Mapping[str, Any]] | None,
    events_arr,
) -> EventAlignment:
    if not event_manifest or events_arr is None:
        return EventAlignment((), ())

    student_pos_by_key: dict[AnchorKey, int] = {}
    for anchor in convo.anchors:
        token_pos = int(anchor.token_pos)
        if anchor.unreachable or anchor.truncated or token_pos < 0 or token_pos >= convo.length:
            continue
        student_pos_by_key[AnchorKey.from_anchor(anchor)] = token_pos

    positions: list[int] = []
    teacher_rows: list[int] = []
    for entry in event_manifest:
        key = AnchorKey.from_entry(entry)
        if key not in student_pos_by_key:
            continue
        row = int(entry.get("row", -1))
        if row < 0 or row >= events_arr.shape[0]:
            continue
        positions.append(student_pos_by_key[key])
        teacher_rows.append(row)

    return EventAlignment(tuple(positions), tuple(teacher_rows))