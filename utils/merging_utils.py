from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import zstandard
import hdf5plugin
import h5py
import os

_worker_handles = {}


def _ensure_handles(dataset_path, teacher_names):
    if _worker_handles.get('path') != dataset_path:
        for f in _worker_handles.get('files', {}).values():
            f.close()
        _worker_handles['path'] = dataset_path
        _worker_handles['files'] = {name: h5py.File(os.path.join(dataset_path, f"{name}.hdf5"), 'r') for name in teacher_names}
    return _worker_handles['files']


def _merge_one_convo(dataset_path, teacher_weights, convo_key):
    files = _ensure_handles(dataset_path, list(teacher_weights.keys()))

    teacher_convos = []
    max_len = 0
    first_sha = ''
    for name, weight in teacher_weights.items():
        f = files[name]
        if convo_key not in f:
            continue
        g = f[convo_key]
        if 'dense_distributions' not in g:
            continue
        distr = np.array(g['dense_distributions'][:], dtype=np.float32)
        convo_len = distr.shape[0]
        teacher_convos.append((distr, convo_len, weight))
        max_len = max(max_len, convo_len)
        if not first_sha:
            first_sha = g.attrs.get('content_sha', '')

    if not teacher_convos:
        return None

    merged = np.zeros((max_len, 256), dtype=np.float32)
    total_w = np.zeros(max_len, dtype=np.float32)
    for distr, length, weight in teacher_convos:
        merged[:length] += weight * distr
        total_w[:length] += weight
    has_data = total_w > 0
    merged[has_data] /= total_w[has_data, np.newaxis]

    dense = merged.astype(np.float16)
    cctx = zstandard.ZstdCompressor(level=1)
    compressed = cctx.compress(dense.tobytes())
    return convo_key, compressed, first_sha, dense.shape


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
    with h5py.File(merged_path, 'w') as merged:
        pbar = tqdm(total=len(convo_keys), desc=f"  {label}", leave=False, smoothing=0.06)
        for result in Parallel(n_jobs=num_workers, backend='loky', return_as='generator_unordered', batch_size=50)(
            delayed(_merge_one_convo)(dataset_path, teacher_weights, key) for key in convo_keys
        ):
            if result is not None:
                key, compressed, sha, shape = result
                g = merged.create_group(key)
                ds = g.create_dataset('dense_distributions', shape=shape, dtype=np.float16, chunks=shape, **hdf5plugin.Zstd(clevel=1))
                ds.id.write_direct_chunk((0, 0), compressed)
                g.attrs['content_sha'] = sha
                g.attrs['convo_len'] = shape[0]
            pbar.update(1)
        pbar.close()
        merged.attrs['teacher_names'] = names_str
        merged.attrs['teacher_weights'] = weights_str


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
    return {name: w / total for name, w in raw.items()}


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
