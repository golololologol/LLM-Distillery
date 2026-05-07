from classes.args import get_config, load_teacher_configs, load_student_config, PipelineConfig, TeacherConfig, StudentConfig
from classes.preprocessing import ModelParticipant
from classes.data_manager import H5DataManager
from classes.dataset_processing import DatasetProcessor
from classes.student.model import train_worker
from classes.student.distributed import find_free_port
from utils.dataset_utils import sync_dataset
from utils.merging_utils import resolve_train_target
from utils.inference_utils import num_gpus
from classes.paths import Paths
from classes.model_runtime import build_participant, build_teacher_participants, build_tokenizer_formatter
from kernels import precompile_training_kernel
from classes.inference.base import get_backend
from tqdm import tqdm
from typing import Any, cast
import torch.multiprocessing as mp
import torch
import logging
import os
import re

logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)

_HF_REPO_PATTERN = re.compile(r'^[a-zA-Z0-9._-]+(/[a-zA-Z0-9._-]+)?$')

def _is_hf_repo_id(path: str) -> bool:
    return bool(_HF_REPO_PATTERN.match(path))


def _section_header(title: str):
    padding = "═" * (65 - len(title))
    print(f"\n═══ {title} {padding}")


def _resolve_model_path(path: str) -> str:
    if os.path.isdir(path):
        return path
    if _is_hf_repo_id(path.split(":")[0]):
        from huggingface_hub import snapshot_download
        parts = path.split(":", 1)
        repo_id = parts[0]
        revision = parts[1] if len(parts) > 1 else None
        return snapshot_download(repo_id, revision=revision)
    raise FileNotFoundError(f"Model directory not found: {path}")


def _collection_params_hash(
    context_len: int,
    temperature: float,
    save_roles: list[str],
    chat_template: str | None = None,
    render_options: dict | None = None,
    segment_handling: dict | None = None,
    specials_hash: str | None = None,
    event_extensions: list | None = None,
) -> str:
    import hashlib
    import json
    opts_str = json.dumps(render_options or {}, sort_keys=True, separators=(",", ":"))
    handling_str = json.dumps(segment_handling or {}, sort_keys=True, separators=(",", ":"))
    tpl_sha = hashlib.sha256((chat_template or "").encode("utf-8")).hexdigest()[:16]
    sp_str = specials_hash or ""
    ext_str = json.dumps(list(event_extensions or []), separators=(",", ":"))
    key = f"{context_len}|{temperature}|{','.join(sorted(save_roles))}|v8.1|{opts_str}|{tpl_sha}|{handling_str}|{sp_str}|{ext_str}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def _discover_all_teachers(teacher_configs: list[tuple[str, TeacherConfig]], dataset_dir: str) -> dict[str, TeacherConfig | None]:
    """Union of teacher configs and existing HDF5s. Returns {name: config_or_None}."""
    teachers: dict[str, TeacherConfig | None] = {name: cfg for name, cfg in teacher_configs}
    for name in _discover_teachers(dataset_dir):
        if name not in teachers:
            teachers[name] = None  # orphan HDF5, no config
    return teachers


def sync_teacher(teacher_name: str, teacher_config: TeacherConfig | None, samples: list[dict], cache_path: str, config: PipelineConfig) -> list[int]:
    """Sync an HDF5 file against samples. Returns list of sample indices needing collection."""
    hdf5_path = os.path.join(cache_path, f"{teacher_name}.hdf5")
    if not os.path.exists(hdf5_path):
        if teacher_config is None or not teacher_config.can_collect:
            return []  # no cache, can't collect - nothing to do
        print(f"  No existing cache, full collection needed ({len(samples)} samples)")
        return list(range(len(samples)))

    processor = DatasetProcessor(config)
    save_roles = processor.save_roles
    data_manager = H5DataManager(cache_path, teacher_name=teacher_name, auto_approve=config.auto_approve)
    try:
        if teacher_config is not None and teacher_config.can_collect:
            if teacher_config.context_len is None:
                raise ValueError(f"Teacher {teacher_name} is collectable but context_len is missing")
            context_len = teacher_config.context_len
            temperature = teacher_config.temperature
            # Compute specials_hash if a specials map is configured.
            sp_hash = None
            specials_map_dict = teacher_config.effective_specials() or None
            if specials_map_dict:
                from classes.event_channel import EventChannelSpec
                from classes.event_vocab import EventVocab
                extensions = tuple(getattr(config, "event_extensions", []) or [])
                sp_hash = EventChannelSpec.hash_for_config(specials_map_dict, extensions)
            params_hash = _collection_params_hash(
                context_len, temperature, list(save_roles),
                chat_template=teacher_config.resolve_chat_template(),
                render_options=teacher_config.render_options,
                segment_handling=teacher_config.effective_segments(),
                specials_hash=sp_hash,
                event_extensions=list(getattr(config, "event_extensions", []) or []),
            )
            stored_hash = data_manager.get_dataset_attr('params_hash')
            if stored_hash is not None and stored_hash != params_hash:
                stale_ids = list(data_manager.get_available_shas().keys())
                if stale_ids:
                    data_manager.delete_ids(
                        stale_ids,
                        reason=f"Collection parameters changed (temperature/context_len/save_roles/specials)"
                    )
            data_manager.set_dataset_attr('params_hash', params_hash)
            data_manager.set_dataset_attr('context_len', context_len)
            data_manager.set_dataset_attr('temperature', teacher_config.temperature)
            data_manager.set_dataset_attr('save_roles', config.save_roles)
            data_manager.set_dataset_attr('segment_handling', teacher_config.effective_segments())
            data_manager.set_dataset_attr('model_name', teacher_name)
            if sp_hash is not None:
                data_manager.set_dataset_attr('specials_hash', sp_hash)
                data_manager.set_dataset_attr(
                    'event_alphabet',
                    list(EventVocab(extensions=tuple(getattr(config, "event_extensions", []) or [])).slots),
                )

        hdf5_shas = data_manager.get_available_shas()
        segment_handling = teacher_config.effective_segments() if teacher_config is not None else None
        sync_result = sync_dataset(samples, save_roles, hdf5_shas)
        print(f"  Sync: {len(sync_result.to_collect)} to collect, {len(sync_result.to_delete)} to delete, {len(sync_result.to_reindex)} to reindex")

        is_orphan = teacher_config is None or not teacher_config.can_collect
        if is_orphan and sync_result.to_delete and len(sync_result.to_delete) == len(hdf5_shas) and len(hdf5_shas) > 0:
            raise ValueError(
                f"All {len(hdf5_shas)} entries in {teacher_name}.hdf5 would be deleted - "
                f"none match the current text dataset. This dataset has no teacher config to re-collect. "
                f"Check that dataset_path is correct, or remove the HDF5, or set train_on explicitly."
            )

        data_manager.sync(sync_result.to_delete, sync_result.to_reindex)

        merge_weight = teacher_config.merge_weight if teacher_config is not None else 1.0
        data_manager.set_dataset_attr('merge_weight', merge_weight)

        data_manager.set_dataset_attr('layout_version', 4)

        return sync_result.to_collect
    finally:
        data_manager.close()


def collect_teacher(teacher_name: str, teacher_config: TeacherConfig, to_collect_indices: list[int], samples: list[dict], cache_path: str, config: PipelineConfig, eligibilities: list | None = None):
    """Collect distributions for specific samples. Assumes sync already happened."""
    if teacher_config.model_path is None:
        raise ValueError(f"Teacher {teacher_name} is missing model_path")
    if teacher_config.context_len is None:
        raise ValueError(f"Teacher {teacher_name} is missing context_len")
    if teacher_config.backend_type is None:
        raise ValueError(f"Teacher {teacher_name} is missing backend_type")
    context_len = teacher_config.context_len
    processor = DatasetProcessor(config)
    temperature = teacher_config.temperature

    if eligibilities is not None:
        before = len(to_collect_indices)
        to_collect_indices = [i for i in to_collect_indices if teacher_name in eligibilities[i].eligible_teachers]
        dropped = before - len(to_collect_indices)
        if dropped:
            print(f"  Eligibility filter: dropped {dropped} samples for teacher {teacher_name}")
        if not to_collect_indices:
            print(f"  No eligible samples for teacher {teacher_name} after filter.")
            return

    to_process_samples = [samples[i] for i in to_collect_indices]

    data_manager = H5DataManager(cache_path, teacher_name=teacher_name, auto_approve=config.auto_approve)
    try:
        model_path = _resolve_model_path(teacher_config.model_path)
        _tokenizer, formatter = build_tokenizer_formatter(model_path, teacher_config)
        segment_handling = teacher_config.effective_segments()
        processed = processor.preprocess(to_process_samples, formatter, context_len, segment_handling=segment_handling)
        del _tokenizer, formatter

        total_tokens = sum(p.length for p in processed)
        content_bytes = sum(e - s for p in processed for s, e in p.byte_ranges())
        print(f"  {len(processed)} samples, {total_tokens:,} tokens, {content_bytes:,} content bytes")

        print(f"  Loading model: {teacher_config.model_path}")
        backend_cls = cast(Any, get_backend(teacher_config.backend_type))
        specials_config = teacher_config.effective_specials() or None
        event_extensions = list(getattr(config, "event_extensions", []) or [])
        backend = backend_cls(
            model_path=model_path,
            context_len=context_len,
            marg_chunk_size=config.marg_chunk_inference,
            temperature=temperature,
            specials_config=specials_config,
            event_extensions=event_extensions,
            **teacher_config.backend_params,
        )
        backend.start(distribution_writer=data_manager.writer)
        try:
            pbar = tqdm(total=len(processed), desc=f"  Collecting", leave=True, smoothing=0.06)
            def update_progress(n: int) -> None:
                pbar.update(n)
            backend.process_chunk(processed, progress_callback=update_progress)
            pbar.close()
        finally:
            backend.stop()
    finally:
        data_manager.close()


def _build_participants(all_teachers: dict, student_config: StudentConfig | None) -> list[ModelParticipant]:
    parts = build_teacher_participants(all_teachers, _resolve_model_path)
    if student_config is not None:
        parts.append(build_participant(
            name="__student__",
            role="student",
            model_path=_resolve_model_path(student_config.model_path),
            config=student_config,
        ))
    return parts


def collect_for_dataset(dataset_path: str, cache_path: str, all_teachers: dict[str, TeacherConfig | None], student_config: StudentConfig | None, config: PipelineConfig, label: str):
    """Sync and collect for all teachers."""
    processor = DatasetProcessor(config)
    samples = processor.load_samples(dataset_path)
    _section_header(f"{label} Dataset")
    print(f"{len(samples)} samples from {dataset_path}\n")

    print("Computing sample eligibility...")
    participants = _build_participants(all_teachers, student_config)
    eligibilities = processor.compute_eligibility(samples, participants)
    globally_dropped = sum(1 for e in eligibilities if not e.eligible_teachers)
    if globally_dropped:
        print(f"  {globally_dropped} samples have no eligible teacher and will be skipped globally.\n")

    for teacher_name, teacher_config in all_teachers.items():
        print(f"┌─ {teacher_name} " + "─" * (56 - len(teacher_name)) + "┐")
        to_collect = sync_teacher(teacher_name, teacher_config, samples, cache_path, config)
        if not to_collect:
            print("  Nothing to collect, skipping.")
        elif teacher_config is not None and teacher_config.can_collect:
            collect_teacher(teacher_name, teacher_config, to_collect, samples, cache_path, config, eligibilities=eligibilities)
        else:
            print(f"  {len(to_collect)} samples need collection but no teacher backend available - using existing data only.")
        print("└" + "─" * (59) + "┘\n")


def _discover_teachers(dataset_path: str) -> list[str]:
    if not os.path.isdir(dataset_path):
        return []
    return sorted(
        os.path.splitext(f)[0]
        for f in os.listdir(dataset_path)
        if f.endswith('.hdf5') and not f.startswith('_')
    )


def main():
    config, validate_only, collect_only, train_only = get_config()

    try:
        teacher_configs = load_teacher_configs(config.teacher_configs_path)
    except FileNotFoundError:
        if not train_only:
            raise
        teacher_configs = []

    student_config = load_student_config(config.student_config_path)

    paths = Paths(config.cache_folder)

    all_teachers = _discover_all_teachers(teacher_configs, paths.dataset)

    collectable = [name for name, conf in all_teachers.items() if conf is not None and conf.can_collect]
    metadata_only = [name for name, conf in all_teachers.items() if conf is not None and not conf.can_collect]
    orphans = [name for name, conf in all_teachers.items() if conf is None]

    print(f"Teachers with backends: {', '.join(collectable) if collectable else 'none'}")
    if metadata_only:
        print(f"Metadata-only configs: {', '.join(metadata_only)}")
    if orphans:
        print(f"Orphan HDF5s (no config): {', '.join(orphans)}")

    if config.train_on != "all":
        selected = config.train_on if isinstance(config.train_on, list) else [config.train_on]
        working_teachers = {n: all_teachers[n] for n in selected if n in all_teachers}
        missing = [n for n in selected if n not in all_teachers]
        if missing:
            raise ValueError(f"train_on teachers not found (no config and no HDF5): {missing}. Available: {sorted(all_teachers.keys())}")
        if not train_only:
            for name in selected:
                cfg = all_teachers.get(name)
                hdf5_exists = os.path.exists(os.path.join(paths.dataset, f"{name}.hdf5"))
                if not hdf5_exists and (cfg is None or not cfg.can_collect):
                    raise ValueError(f"train_on teacher '{name}' has no HDF5 data and no backend to collect with.")
    else:
        working_teachers = all_teachers

    if validate_only:
        print("Config validation passed.")
        print(f"  Pipeline config: OK")
        print(f"  Working teachers: {sorted(working_teachers.keys())}")
        print(f"  Student: {student_config.model_path}")
        print(f"  Loss: {config.loss_type}, Optimizer: {config.optimizer}, Scheduler: {config.lr_scheduler}")
        print(f"  Precision: {config.training_precision}, Strategy: {config.training_strategy}")
        return

    if not train_only:
        collect_for_dataset(config.dataset_path, paths.dataset, working_teachers, student_config, config, "Training")
        collect_for_dataset(config.validation_dataset_path, paths.dataset_validation, working_teachers, student_config, config, "Validation")
        print("Collection complete.")
        if collect_only:
            return

    available_teachers = _discover_teachers(paths.dataset)
    if config.train_on != "all":
        available_teachers = [t for t in available_teachers if t in working_teachers]

    if not available_teachers:
        raise ValueError(f"No teacher data found in {paths.dataset}. Run collection first.")

    train_target = resolve_train_target(config, available_teachers, working_teachers, paths)

    torch.cuda.empty_cache()
    if config.multi_gpu and config.training_strategy in ("ddp", "fsdp2"):
        world_size = num_gpus()
    else:
        world_size = 1
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(find_free_port())

    _section_header("Training")
    print(f"Target: {train_target}")
    print(f"Student: {student_config.model_path}")
    print(f"Loss: {config.loss_type} | Optimizer: {config.optimizer} | Scheduler: {config.lr_scheduler}")
    print(f"Precision: {config.training_precision} | Strategy: {config.training_strategy} | GPUs: {world_size}")
    
    precompile_training_kernel(config.loss_type)

    if world_size > 1:
        cast(Any, mp).spawn(train_worker, args=(world_size, config, student_config, paths, train_target), nprocs=world_size, join=True)
    else:
        train_worker(0, world_size, config, student_config, paths, train_target)
    print("Training complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
