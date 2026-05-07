from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import zstandard
import hdf5plugin
import json
import h5py
import logging
import os
from typing import Any, cast

from classes.segment_alignment import AnchorKey, SegmentKey

log = logging.getLogger(__name__)

_worker_handles = {}


def _decode_event_alphabet(ea) -> tuple | None:
    """Tolerantly decode an event_alphabet attribute.

    Older caches stored it as a JSON string; current caches store it as an
    ndarray of strings. Both forms must round-trip to the same tuple.
    """
    if ea is None:
        return None
    if isinstance(ea, (bytes, str)):
        try:
            return tuple(json.loads(ea))
        except Exception:
            return None
    try:
        return tuple(str(x) for x in ea)
    except Exception:
        return None


def _ensure_handles(dataset_path, teacher_names):
    if _worker_handles.get('path') != dataset_path:
        for f in _worker_handles.get('files', {}).values():
            f.close()
        _worker_handles['path'] = dataset_path
        files = {}
        for name in teacher_names:
            path = os.path.join(dataset_path, f"{name}.hdf5")
            f = h5py.File(path, 'r')
            layout = f.attrs.get('layout_version')
            if layout is None or int(layout) < 4:
                f.close()
                raise RuntimeError(f"cache at {path} has layout_version={layout}; expected layout_version>=4")
            files[name] = f
        _worker_handles['files'] = files
    # Verify event_alphabet identity across teachers (when present).
    alphabets = {}
    for name, f in _worker_handles['files'].items():
        ea = f.attrs.get('event_alphabet')
        if ea is not None:
            alphabets[name] = _decode_event_alphabet(ea)
    if alphabets:
        ref = next(iter(alphabets.values()))
        bad = [n for n, a in alphabets.items() if a != ref]
        if bad:
            log.warning(
                f"event_alphabet mismatch across teachers: {bad}. "
                f"Event distributions will not be merged."
            )
            _worker_handles['event_alphabet_mismatch'] = True
        else:
            _worker_handles['event_alphabet_mismatch'] = False
    return _worker_handles['files']


def _log_reject(rejects, reason):
    rejects[reason] = rejects.get(reason, 0) + 1


def _pick_eligible_group(groups, seg_key, convo_key, rejects):
    # groups: {hash: [contributor]}, contributor = (teacher, weight, arr, truncated, byte_count)
    ranked = sorted(
        groups.items(),
        key=lambda kv: (-len(kv[1]), -sum(c[1] for c in kv[1]), min(c[0] for c in kv[1])),
    )
    winning_hash, winners = ranked[0]
    tied = [h for h, m in groups.items()
            if h != winning_hash and len(m) == len(winners) and sum(c[1] for c in m) == sum(c[1] for c in winners)]
    for h, members in groups.items():
        if h == winning_hash:
            continue
        reason = "tie_lost_lower_weight" if h in tied or len(members) == len(winners) else "hash_mismatch"
        for _c in members:
            _log_reject(rejects, reason)
    return winning_hash, winners


def _merge_segment(winners, seg_key, convo_key, winning_hash, is_tool_call, rejects):
    # winners: list of (teacher, weight, arr, truncated, byte_count, seg_meta)
    if is_tool_call:
        max_bc = max(c[4] for c in winners)
        winners, rejected = [c for c in winners if c[4] == max_bc], [c for c in winners if c[4] != max_bc]
        for _ in rejected:
            _log_reject(rejects, "tool_call_length_mismatch")
        L = max_bc
        result_truncated = all(c[3] for c in winners)
    else:
        L = max(c[4] for c in winners)
        winners, rejected = [c for c in winners if c[4] >= L or c[3]], [c for c in winners if c[4] < L and not c[3]]
        for _ in rejected:
            _log_reject(rejects, "non_truncated_short")
        result_truncated = not any((c[4] == L and not c[3]) for c in winners)
    if not winners:
        return None, None

    merged = np.zeros((L, 256), dtype=np.float32)
    weight_sum = np.zeros(L, dtype=np.float32)
    for _, w, arr, _, bc, _ in winners:
        n = min(bc, L)
        merged[:n] += w * arr[:n]
        weight_sum[:n] += w
    has = weight_sum > 0
    merged[has] /= weight_sum[has, np.newaxis]

    # Carry the full segment key in the merged manifest.
    sample_meta = winners[0][5]
    entry = {
        "msg_idx": seg_key.msg_idx,
        "segment_idx": seg_key.segment_idx,
        "type": seg_key.type,
        "tool_call_idx": seg_key.tool_call_idx,
        "byte_count": L,
        "canonical_target_hash": winning_hash,
        "truncated": result_truncated,
        "handling": sample_meta.get("handling", "active"),
        "role": sample_meta.get("role", ""),
    }
    return merged.astype(np.float16), entry


def _merge_one_convo(dataset_path, teacher_weights, convo_key):
    files = _ensure_handles(dataset_path, list(teacher_weights.keys()))
    rejects: dict[str, int] = {}

    unified_segments = []
    seen = set()
    teacher_contribs = {}  # name -> {seg_key: (arr, hash, truncated, byte_count, meta)}
    teacher_events: dict[str, dict] = {}  # name -> {anchor_key: (row_arr, supported_mask, meta)}
    unified_anchors: list = []
    seen_anchors: set = set()
    event_alphabet: tuple | None = None
    first_sha = None
    union_supported_mask = 0
    skip_event = False

    for name in teacher_weights:
        group = files[name].get(convo_key)
        if group is None or 'dense_distributions' not in group:
            continue
        if first_sha is None:
            first_sha = group.attrs.get('content_sha', '')

        manifest = json.loads(group.attrs['segment_manifest'])
        data = np.array(group['dense_distributions'][:], dtype=np.float32)

        cursor = 0
        segs = {}
        for seg in manifest:
            key = SegmentKey.from_entry(seg)
            bc = seg["byte_count"]
            segs[key] = (
                data[cursor:cursor + bc],
                seg.get("canonical_target_hash", ""),
                bool(seg.get("truncated", False)),
                bc,
                seg,
            )
            cursor += bc
            if key not in seen:
                unified_segments.append(key)
                seen.add(key)
        teacher_contribs[name] = segs

        # Event channel
        if not skip_event and 'event_distributions' in group and 'event_manifest' in group.attrs:
            ev_arr = np.array(group['event_distributions'][:], dtype=np.float32)
            ev_manifest = json.loads(group.attrs['event_manifest'])
            sup_mask = int(group.attrs.get('supported_mask', 0))
            ea = group.attrs.get('event_alphabet')
            if ea is not None:
                ea_t = _decode_event_alphabet(ea)
                if event_alphabet is None:
                    event_alphabet = ea_t
                elif event_alphabet != ea_t:
                    # Mismatch: discard all event data for this conversation
                    log.warning(
                        f"event_alphabet mismatch in convo {convo_key}: {ea_t} vs {event_alphabet}. "
                        f"Skipping event distributions."
                    )
                    skip_event = True
                    teacher_events.clear()
                    unified_anchors.clear()
                    seen_anchors.clear()
                    event_alphabet = None
                    union_supported_mask = 0
                    continue
            ev_dict = {}
            for ent in ev_manifest:
                a_key = AnchorKey.from_entry(ent)
                row = ent.get("row")
                if row is None or row < 0 or row >= ev_arr.shape[0]:
                    continue
                ev_dict[a_key] = (ev_arr[row], sup_mask, ent)
                if a_key not in seen_anchors:
                    unified_anchors.append(a_key)
                    seen_anchors.add(a_key)
            teacher_events[name] = ev_dict
            union_supported_mask |= sup_mask

    if not teacher_contribs:
        return None

    chunks = []
    merged_manifest = []
    total_segments = 0

    for seg_key in unified_segments:
        groups = {}
        for name, segs in teacher_contribs.items():
            if seg_key not in segs:
                continue
            arr, h, trunc, bc, meta = segs[seg_key]
            groups.setdefault(h, []).append((name, teacher_weights[name], arr, trunc, bc, meta))

        if not groups:
            continue
        total_segments += sum(len(v) for v in groups.values())

        winning_hash, winners = _pick_eligible_group(groups, seg_key, convo_key, rejects)

        merged_seg, entry = _merge_segment(
            winners, seg_key, convo_key, winning_hash,
            seg_key.type == "tool_call", rejects,
        )
        if merged_seg is None:
            continue
        chunks.append(merged_seg)
        merged_manifest.append(entry)

    merged = np.concatenate(chunks) if chunks else np.zeros((0, 256), dtype=np.float16)
    compressed = zstandard.ZstdCompressor(level=1).compress(merged.tobytes())

    # Event-channel merge: per-slot mean over teachers whose supported_mask
    # has the slot bit set, weighted by teacher merge_weight. Renormalise the
    # final row over the union-supported slots only. Residual is NEVER routed
    # to E_OTHER_SPECIAL (plan §6).
    merged_events = None
    merged_event_manifest: list[dict] = []
    if event_alphabet is not None and unified_anchors:
        E = len(event_alphabet)
        slot_idx = np.arange(E, dtype=np.int64)
        rows = []
        for a_key in unified_anchors:
            num = np.zeros(E, dtype=np.float64)
            wsum = np.zeros(E, dtype=np.float64)
            contrib_meta = None
            for name, ev_dict in teacher_events.items():
                if a_key not in ev_dict:
                    continue
                row, mask, meta = ev_dict[a_key]
                if contrib_meta is None:
                    contrib_meta = meta
                w = teacher_weights[name]
                supported = ((mask >> slot_idx) & 1).astype(bool)
                num[supported] += w * row[supported]
                wsum[supported] += w
            if contrib_meta is None:
                continue
            slot_mean = np.zeros(E, dtype=np.float64)
            has = wsum > 0
            slot_mean[has] = num[has] / wsum[has]
            row_total = slot_mean.sum()
            if row_total > 0:
                slot_mean = slot_mean / row_total
            row_idx = len(rows)
            rows.append(slot_mean.astype(np.float16))
            merged_event_manifest.append({
                "msg_idx": a_key.msg_idx,
                "segment_idx": a_key.segment_idx,
                "anchor_side": a_key.anchor_side,
                "row": row_idx,
                "unreachable": bool(contrib_meta.get("unreachable", False)),
                "truncated": bool(contrib_meta.get("truncated", False)),
            })
        merged_events = (
            np.stack(rows, axis=0) if rows else np.zeros((0, E), dtype=np.float16)
        )

    return (
        convo_key, compressed, first_sha, json.dumps(merged_manifest), merged.shape, rejects, total_segments,
        merged_events, json.dumps(merged_event_manifest) if merged_event_manifest else None,
        list(event_alphabet) if event_alphabet is not None else None,
        union_supported_mask,
    )


def merge_teacher_files(dataset_path: str, teacher_weights: dict[str, float], label: str = "Merging"):
    merged_path = os.path.join(dataset_path, "_merged.hdf5")

    sorted_names = sorted(teacher_weights.keys())
    sorted_weights = [teacher_weights[name] for name in sorted_names]
    names_str = ",".join(sorted_names)
    weights_str = ",".join(f"{weight:.6f}" for weight in sorted_weights)

    source_paths = []
    for name in teacher_weights:
        path = os.path.join(dataset_path, f"{name}.hdf5")
        if not os.path.exists(path):
            raise FileNotFoundError(f"Teacher data not found: {path}")
        source_paths.append(path)

    if os.path.exists(merged_path):
        merged_mtime = os.path.getmtime(merged_path)
        sources_fresh = all(os.path.getmtime(p) <= merged_mtime for p in source_paths)
        if sources_fresh:
            try:
                with h5py.File(merged_path, 'r') as merged_file:
                    if merged_file.attrs.get('teacher_names', '') == names_str and merged_file.attrs.get('teacher_weights', '') == weights_str:
                        print(f"  {label}: up-to-date, skipping")
                        return
            except Exception:
                pass

    # Discover all convo keys across teachers
    convo_keys_set = set()
    for name in teacher_weights:
        with h5py.File(os.path.join(dataset_path, f"{name}.hdf5"), 'r') as f:
            convo_keys_set.update(k for k in f.keys() if k.startswith('convo_'))
    convo_keys = sorted(convo_keys_set)

    num_workers = min(os.cpu_count() or 4, 8)

    # Write results as they stream from workers
    total_rejects: dict[str, int] = {}
    total_segments = 0
    written_alphabet: list | None = None
    with h5py.File(merged_path, 'w') as merged:
        merged.attrs['layout_version'] = 5
        pbar = tqdm(total=len(convo_keys), desc=f"  {label}", leave=False, smoothing=0.06)
        for result in Parallel(n_jobs=num_workers, backend='loky', return_as='generator_unordered', batch_size=cast(Any, 50))(
            delayed(_merge_one_convo)(dataset_path, teacher_weights, key) for key in convo_keys
        ):
            if result is not None:
                (
                    key, compressed, sha, manifest_json, shape, rejects, seg_count,
                    merged_events, event_manifest_json, event_alphabet, union_mask,
                ) = result
                g = merged.create_group(key)
                zstd = cast(Any, getattr(hdf5plugin, "Zstd"))(clevel=1)
                ds = g.create_dataset('dense_distributions', shape=shape, dtype=np.float16, chunks=shape, **zstd)
                ds.id.write_direct_chunk((0, 0), compressed)
                g.attrs['content_sha'] = sha
                g.attrs['convo_len'] = shape[0]
                if manifest_json:
                    g.attrs['segment_manifest'] = manifest_json
                if merged_events is not None and merged_events.size > 0:
                    g.create_dataset(
                        'event_distributions', data=merged_events,
                        **zstd,
                    )
                    g.attrs['event_count'] = merged_events.shape[0]
                    g.attrs['event_manifest'] = event_manifest_json
                    g.attrs['supported_mask'] = int(union_mask)
                    if event_alphabet is not None:
                        g.attrs['event_alphabet'] = list(event_alphabet)
                        if written_alphabet is None:
                            written_alphabet = event_alphabet
                else:
                    g.attrs['event_count'] = 0
                for r, n in rejects.items():
                    total_rejects[r] = total_rejects.get(r, 0) + n
                total_segments += seg_count
            pbar.update(1)
        pbar.close()
        merged.attrs['teacher_names'] = names_str
        merged.attrs['teacher_weights'] = weights_str
        if written_alphabet is not None:
            merged.attrs['event_alphabet'] = json.dumps(written_alphabet)

    total_rej = sum(total_rejects.values())
    if total_rej:
        parts = ", ".join(f"{r}={n}" for r, n in sorted(total_rejects.items()))
        print(f"  [{label}] merge rejects: {parts} (total {total_rej} of {total_segments} contributions)")
        if total_segments and total_rej / total_segments > 0.10:
            log.warning(
                f"[{label}] high merge reject rate: {total_rej}/{total_segments} "
                f"({100 * total_rej / total_segments:.1f}%). Inspect reasons: {parts}"
            )


def get_merge_weights(teacher_names: list[str], all_teachers: dict, dataset_dir: str) -> dict[str, float]:
    raw = {}
    for name in teacher_names:
        cfg = all_teachers.get(name)
        if cfg is not None:
            raw[name] = cfg.merge_weight
        else:
            hdf5_path = os.path.join(dataset_dir, f"{name}.hdf5")
            weight = 1.0
            if os.path.exists(hdf5_path):
                try:
                    with h5py.File(hdf5_path, 'r') as f:
                        weight = float(f.attrs.get('merge_weight', 1.0))
                except Exception:
                    pass
            raw[name] = weight
    total = sum(raw.values())
    return {name: weight / total for name, weight in raw.items()}


def resolve_train_target(config, available_teachers: list[str], all_teachers: dict, paths) -> str:
    if config.train_on != "all":
        selected = config.train_on if isinstance(config.train_on, list) else [config.train_on]
        for name in selected:
            if name not in available_teachers:
                raise ValueError(f"train_on teacher '{name}' not found. Available: {available_teachers}")
        if len(selected) == 1:
            return selected[0]
        teacher_weights = get_merge_weights(selected, all_teachers, paths.dataset)
    else:
        if len(available_teachers) == 1:
            return available_teachers[0]
        selected = available_teachers
        teacher_weights = get_merge_weights(selected, all_teachers, paths.dataset)

    padding = "═" * (65 - len("Merging"))
    print(f"\n═══ Merging {padding}")
    for name, w in teacher_weights.items():
        print(f"  {name}: {w:.3f}")
    merge_teacher_files(paths.dataset, teacher_weights, label="Merging train")
    merge_teacher_files(paths.dataset_validation, teacher_weights, label="Merging val")
    return "_merged"
