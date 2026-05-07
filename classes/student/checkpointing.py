import dataclasses
import gc
import time
import torch
import torch.distributed as dist
import numpy as np
import shutil
import glob
import re
import os

from classes.data_classes import TrainingState


def _rename_with_retry(src, dst, retries=5, delay=0.5):
    for attempt in range(retries):
        try:
            os.rename(src, dst)
            return
        except PermissionError:
            if attempt == retries - 1:
                raise
            gc.collect()
            time.sleep(delay * (1.5 ** attempt))


def save_model(model, tokenizer, save_dir, strategy, rank, world_size):
    save_dir_tmp = save_dir + ".tmp"
    if rank == 0 and os.path.exists(save_dir_tmp):
        shutil.rmtree(save_dir_tmp)

    if strategy == "fsdp2":
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, StateDictOptions
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_sd = get_model_state_dict(model, options=opts)
        if rank == 0:
            os.makedirs(save_dir_tmp, exist_ok=True)
            model.save_pretrained(save_dir_tmp, state_dict=model_sd)
            tokenizer.save_pretrained(save_dir_tmp)
        if world_size > 1:
            dist.barrier()
    else:
        if rank == 0:
            os.makedirs(save_dir_tmp, exist_ok=True)
            raw_model = model.module if hasattr(model, "module") else model
            raw_model.save_pretrained(save_dir_tmp)
            tokenizer.save_pretrained(save_dir_tmp)
        if world_size > 1:
            dist.barrier()

    if rank == 0:
        if os.path.exists(save_dir):
            old_dir = save_dir + ".old"
            if os.path.exists(old_dir):
                shutil.rmtree(old_dir)
            _rename_with_retry(save_dir, old_dir)
        _rename_with_retry(save_dir_tmp, save_dir)
        if os.path.exists(save_dir + ".old"):
            shutil.rmtree(save_dir + ".old")
    if world_size > 1:
        dist.barrier()


def save_training_state(model, tokenizer, optimizer, lr_scheduler, save_dir, strategy, rank, world_size, metadata: TrainingState):
    save_dir_tmp = save_dir + ".tmp"
    if rank == 0 and os.path.exists(save_dir_tmp):
        shutil.rmtree(save_dir_tmp)

    schedule_free = hasattr(optimizer, 'eval')
    if schedule_free:
        optimizer.eval()

    if strategy == "fsdp2":
        from torch.distributed.checkpoint.state_dict import get_model_state_dict, get_optimizer_state_dict, StateDictOptions
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        model_sd = get_model_state_dict(model, options=opts)
        opt_state = get_optimizer_state_dict(model, optimizer, options=opts)
        if rank == 0:
            os.makedirs(save_dir_tmp, exist_ok=True)
            model.save_pretrained(save_dir_tmp, state_dict=model_sd)
            tokenizer.save_pretrained(save_dir_tmp)
        if world_size > 1:
            dist.barrier()
    else:
        if rank == 0:
            os.makedirs(save_dir_tmp, exist_ok=True)
            raw_model = model.module if hasattr(model, "module") else model
            raw_model.save_pretrained(save_dir_tmp)
            tokenizer.save_pretrained(save_dir_tmp)
        opt_state = optimizer.state_dict()

    if schedule_free:
        optimizer.train()

    if rank == 0:
        state = {
            "optimizer_state_dict": opt_state,
            "scheduler_state_dict": lr_scheduler.state_dict() if lr_scheduler else None,
            "rng_torch": torch.random.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state(),
            "rng_numpy": np.random.get_state(),
            **dataclasses.asdict(metadata),
        }
        torch.save(state, os.path.join(save_dir_tmp, "training_state.pt"))
        if os.path.exists(save_dir):
            old_dir = save_dir + ".old"
            if os.path.exists(old_dir):
                shutil.rmtree(old_dir)
            _rename_with_retry(save_dir, old_dir)
        _rename_with_retry(save_dir_tmp, save_dir)
        if os.path.exists(save_dir + ".old"):
            shutil.rmtree(save_dir + ".old")
    if world_size > 1:
        dist.barrier()


def load_training_state(checkpoint_dir, model, optimizer, lr_scheduler, strategy, rank, world_size):
    try:
        state = torch.load(os.path.join(checkpoint_dir, "training_state.pt"), map_location="cpu", weights_only=False)
    except Exception as e:
        raise RuntimeError(
            f"Failed to load training state from {checkpoint_dir}: {e}\n"
            f"The checkpoint may be corrupt. Try resuming from a different checkpoint."
        ) from e

    if strategy == "fsdp2":
        from torch.distributed.checkpoint.state_dict import set_optimizer_state_dict, StateDictOptions
        opts = StateDictOptions(full_state_dict=True, cpu_offload=True)
        set_optimizer_state_dict(model, optimizer, state["optimizer_state_dict"], options=opts)
    else:
        optimizer.load_state_dict(state["optimizer_state_dict"])

    if hasattr(optimizer, 'train'):
        optimizer.train()

    if lr_scheduler and state.get("scheduler_state_dict"):
        lr_scheduler.load_state_dict(state["scheduler_state_dict"])

    if "rng_torch" in state:
        torch.random.set_rng_state(state["rng_torch"])
    if "rng_cuda" in state:
        torch.cuda.set_rng_state(state["rng_cuda"])
    if "rng_numpy" in state:
        np.random.set_state(state["rng_numpy"])

    return TrainingState(
        num_trained=state["num_trained"],
        epoch=state["epoch"],
        next_accum=state["next_accum"],
        next_val=state["next_val"],
        next_save=state["next_save"],
        next_state_save=state.get("next_state_save"),
        wandb_run_id=state.get("wandb_run_id"),
        best_val_loss=state.get("best_val_loss"),
    )


def resolve_checkpoint_dir(resume_from, states_dir, model_name):
    if resume_from == "latest":
        pattern = os.path.join(states_dir, f"{model_name}_training_state_step_*")
        dirs = [d for d in glob.glob(pattern) if os.path.isdir(d) and not d.endswith(".tmp") and not d.endswith(".old")]
        if not dirs:
            raise FileNotFoundError(f"No checkpoints found matching {pattern}")
        def _extract_step(path):
            match = re.search(r"_step_(\d+)$", path)
            return int(match.group(1)) if match else -1
        return max(dirs, key=_extract_step)
    elif resume_from == "best":
        best_dir = os.path.join(states_dir, f"{model_name}_training_state_step_best")
        if not os.path.isdir(best_dir):
            raise FileNotFoundError(f"No 'best' checkpoint found at {best_dir}")
        if not os.path.isfile(os.path.join(best_dir, "training_state.pt")):
            raise FileNotFoundError(f"No training_state.pt found in {best_dir}")
        return best_dir
    else:
        if not os.path.isfile(os.path.join(resume_from, "training_state.pt")):
            raise FileNotFoundError(f"No training_state.pt found in {resume_from}")
        return resume_from


def cleanup_incomplete_checkpoints(states_dir):
    if not os.path.isdir(states_dir):
        return
    for entry in os.scandir(states_dir):
        if entry.is_dir() and entry.name.endswith(".old"):
            shutil.rmtree(entry.path)
        elif entry.is_dir() and entry.name.endswith(".tmp"):
            shutil.rmtree(entry.path)


def rotate_checkpoints(base_dir, pattern_prefix, keep_last_n):
    pattern = os.path.join(base_dir, pattern_prefix + "*")
    dirs = [d for d in glob.glob(pattern) if os.path.isdir(d) and not d.endswith(".tmp") and not d.endswith(".old")]
    def _extract_step(path):
        match = re.search(r"_step_(\d+)$", path)
        return int(match.group(1)) if match else -1
    dirs = [d for d in dirs if _extract_step(d) >= 0]
    if len(dirs) <= keep_last_n:
        return
    dirs.sort(key=_extract_step)
    for d in dirs[:-keep_last_n]:
        shutil.rmtree(d)
