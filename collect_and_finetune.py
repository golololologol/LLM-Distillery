from classes.args import get_config, load_teacher_configs, load_student_config, PipelineConfig, TeacherConfig
from classes.preprocessing import preprocess_samples
from classes.data_manager import H5DataManager
from classes.student.model import train_worker, resolve_strategy
from classes.student.distributed import find_free_port
from utils.dataset_utils import read_jsonl, sync_dataset
from utils.inference_utils import num_gpus
from classes.paths import Paths
from transformers import AutoTokenizer
from kernels import precompile_training_kernel
from classes.inference.base import get_backend
from tqdm import tqdm
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


def _collection_params_hash(context_len: int, temperature: float, save_roles: list[str]) -> str:
    import hashlib
    key = f"{context_len}|{temperature}|{','.join(sorted(save_roles))}"
    return hashlib.sha256(key.encode()).hexdigest()[:16]


def collect_teacher(teacher_name: str, teacher_config: TeacherConfig, samples: list[dict], cache_path: str, config: PipelineConfig):
    context_len = teacher_config.context_len or config.context_len
    save_roles = set(config.save_roles)

    data_manager = H5DataManager(cache_path, teacher_name=teacher_name, auto_approve=config.auto_approve)
    try:
        data_manager.set_dataset_attr('format_version', '1')

        existing_teachers = data_manager.get_dataset_attr('teachers') or []
        if teacher_name not in existing_teachers:
            data_manager.set_dataset_attr('teachers', existing_teachers + [teacher_name])

        temperature = teacher_config.temperature if teacher_config.temperature is not None else config.collection_temperature
        params_hash = _collection_params_hash(context_len, temperature, list(save_roles))
        stored_hash = data_manager.get_teacher_attr('params_hash')
        params_changed = stored_hash is not None and stored_hash != params_hash

        hdf5_shas = data_manager.get_available_shas()

        if params_changed:
            stale_ids = list(hdf5_shas.keys())
            if stale_ids:
                data_manager.delete_ids(
                    stale_ids,
                    reason=f"[{teacher_name}] Collection parameters changed (temperature/context_len/save_roles)."
                )
            hdf5_shas = {}
            
        data_manager.set_teacher_attr('model_name', teacher_name)
        data_manager.set_teacher_attr('context_len', context_len)
        data_manager.set_teacher_attr('save_roles', config.save_roles)
        data_manager.set_teacher_attr('temperature', temperature)
        data_manager.set_teacher_attr('params_hash', params_hash)

        sync_result = sync_dataset(samples, save_roles, hdf5_shas)
        data_manager.sync(sync_result.to_delete, sync_result.to_reindex)

        if not sync_result.to_collect:
            print(f"  [{teacher_name}] Nothing to collect, skipping.")
            return

        to_process = [samples[i] for i in sync_result.to_collect]

        model_path = _resolve_model_path(teacher_config.model_path)
        tokenizer = AutoTokenizer.from_pretrained(model_path)
        if config.chat_template:
            tokenizer.chat_template = config.chat_template
        processed = preprocess_samples(to_process, tokenizer, context_len, save_roles)
        del tokenizer

        total_tokens = sum(p.length for p in processed)
        content_bytes = sum(e - s for p in processed for s, e in p.content_byte_ranges)
        print(f"  [{teacher_name}] {len(processed)} samples, {total_tokens:,} tokens, {content_bytes:,} content bytes")

        print(f"  [{teacher_name}] Loading teacher model: {teacher_config.model_path}")
        backend_cls = get_backend(teacher_config.backend_type)
        backend = backend_cls(model_path=model_path, context_len=context_len, marg_chunk_size=config.marg_chunk_inference, temperature=temperature, **teacher_config.backend_params)
        backend.start(dm_queue=data_manager.queue)
        try:
            pbar = tqdm(total=len(processed), desc=f"  [{teacher_name}] Collecting", leave=False, smoothing=0.06)
            backend.process_chunk(processed, progress_callback=lambda n: pbar.update(n))
            pbar.close()
        finally:
            backend.stop()
    finally:
        data_manager.close()


def collect_for_dataset(dataset_path: str, cache_path: str, teachers: list[tuple[str, TeacherConfig]], config: PipelineConfig, label: str):
    samples = read_jsonl(dataset_path)
    print(f"\n{label} dataset: {len(samples)} samples, teachers: {[name for name, _ in teachers]}")

    for teacher_name, teacher_config in teachers:
        collect_teacher(teacher_name, teacher_config, samples, cache_path, config)


def determine_train_target(config: PipelineConfig, teachers: list[tuple[str, TeacherConfig]], paths: Paths) -> str:
    teacher_names = [name for name, _ in teachers]

    if config.train_on != "auto":
        selected = config.train_on if isinstance(config.train_on, list) else [config.train_on]

        for name in selected:
            if name not in teacher_names:
                raise ValueError(f"train_on teacher '{name}' not found. Available: {teacher_names}")

        if len(selected) == 1:
            return selected[0]

        selected_teachers = [(name, cfg) for name, cfg in teachers if name in selected]
        return _merge_teachers(selected_teachers, paths, config, label="selected")

    if len(teacher_names) == 1:
        return teacher_names[0]

    return _merge_teachers(teachers, paths, config, label="multi-teacher")


def _merge_teachers(teachers: list[tuple[str, TeacherConfig]], paths: Paths, config: PipelineConfig, label: str) -> str:
    raw_weights = {name: cfg.merge_weight for name, cfg in teachers}
    total = sum(raw_weights.values())
    teacher_weights = {name: w / total for name, w in raw_weights.items()}

    print(f"\nMerging {label} distributions...")
    for name, w in teacher_weights.items():
        print(f"  {name}: {w:.3f}")
    for hdf5_dir in [paths.dataset, paths.dataset_validation]:
        dm = H5DataManager(hdf5_dir, auto_approve=config.auto_approve)
        dm.merge_teachers(teacher_weights)
        dm.close()
    return "_merged"


def main():
    config, validate_only = get_config()
    teachers = load_teacher_configs(config.teacher_configs_path)
    student_config = load_student_config(config.student_config_path)
    print(f"Teachers: {[name for name, _ in teachers]}")

    if validate_only:
        print("Config validation passed.")
        print(f"  Pipeline config: OK")
        print(f"  Teachers: {[name for name, _ in teachers]}")
        print(f"  Student: {student_config.model_path}")
        print(f"  Loss: {config.loss_type}, Optimizer: {config.optimizer}, Scheduler: {config.lr_scheduler}")
        print(f"  Precision: {config.training_precision}, Strategy: {config.training_strategy}")
        return

    paths = Paths(config.cache_folder)

    # Phase 1: Collection
    collect_for_dataset(config.dataset_path, paths.dataset, teachers, config, "Training")
    collect_for_dataset(config.validation_dataset_path, paths.dataset_validation, teachers, config, "Validation")
    print("Collection complete.")

    train_target = determine_train_target(config, teachers, paths)
    print(f"\nTraining on: {train_target}")
    
    precompile_training_kernel(config.loss_type)

    torch.cuda.empty_cache()
    strategy = resolve_strategy(config.training_strategy, config.multi_gpu)
    if config.multi_gpu and strategy in ("ddp", "fsdp2"):
        world_size = num_gpus()
    else:
        world_size = 1
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = str(find_free_port())

    if world_size > 1:
        mp.spawn(train_worker, args=(world_size, config, student_config, paths, train_target), nprocs=world_size, join=True)
    else:
        train_worker(0, world_size, config, student_config, paths, train_target)
    print("Training complete.")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\nInterrupted.")
