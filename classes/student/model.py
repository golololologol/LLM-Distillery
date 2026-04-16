from utils.optimizer_utils import set_optimizer, set_lr_scheduler
from classes.preprocessing import preprocess_samples
from classes.byte_vocab import ByteVocabIndex
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.nn.parallel import DistributedDataParallel as DDP
from classes.data_manager import H5DataManager
from transformers import BitsAndBytesConfig
from utils.dataset_utils import read_jsonl
from classes.args import PipelineConfig, StudentConfig
from classes.data_classes import TrainingState
from classes.losses import Losses, calculate_divergence
from utils.kernel_utils import _try_fused_train
from classes.student.checkpointing import save_model, save_training_state, load_training_state, resolve_checkpoint_dir, cleanup_incomplete_checkpoints, rotate_checkpoints
from classes.student.distributed import setup_distributed, wrap_distributed, get_transformer_layers
from classes.paths import Paths
from pathlib import Path
from tqdm import tqdm
import torch.distributed as dist
import numpy as np
import torch
import wandb
import math
import time
import warnings
import logging
import sys
import os
import gc

# Suppress noisy transformers/torch warnings
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("huggingface_hub").setLevel(logging.ERROR)
warnings.filterwarnings("ignore", message=".*tie_word_embeddings.*")
warnings.filterwarnings("ignore", message=".*use_cache=True.*is incompatible.*")

try:
    from transformers.utils import logging as hf_logging
    hf_logging.disable_progress_bar()
except Exception:
    pass


def train_worker(rank, world_size, config: PipelineConfig, student_config: StudentConfig, paths: Paths, train_target: str):
    """Per-GPU training worker. Called by mp.spawn or directly for single-GPU."""
    torch.cuda.set_device(rank)
    if world_size > 1:
        setup_distributed(rank, world_size)
        torch.set_num_threads(max(1, os.cpu_count() // world_size))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    student = StudentModel(config, student_config, paths)
    try:
        student.train(train_target, rank, world_size)
    finally:
        if world_size > 1 and dist.is_initialized():
            dist.destroy_process_group()


def _build_custom_device_map(model_path, num_gpu0_layers, max_memory):
    from transformers import AutoConfig
    from utils.inference_utils import num_gpus as n_gpus

    num_gpus = n_gpus()
    effective_max_memory = max_memory if len(max_memory) == num_gpus else None

    config = AutoConfig.from_pretrained(model_path)
    layer_names = ["model.embed_tokens"]
    for i in range(config.num_hidden_layers):
        layer_names.append(f"model.layers.{i}")
    layer_names.extend(["model.norm", "lm_head"])

    num_layers = len(layer_names)
    remaining = num_layers - num_gpu0_layers
    per_gpu = math.ceil(remaining / (num_gpus - 1))

    device_map = {}
    idx = 0
    for _ in range(num_gpu0_layers):
        device_map[layer_names[idx]] = 0
        idx += 1

    gpu_id = 1
    while idx < num_layers:
        for _ in range(per_gpu):
            if idx >= num_layers:
                break
            device_map[layer_names[idx]] = gpu_id
            idx += 1
        gpu_id += 1

    return device_map


def _patch_device_hooks(model):
    """Patch Accelerate's AlignDevicesHook to set CUDA device before forward.

    When HF device_map spreads layers across GPUs, Accelerate moves tensors to the
    right device but never calls torch.cuda.set_device(). Triton kernels (Liger)
    launch on torch.cuda.current_device(), so they crash if it doesn't match.
    This patches each hooked module's pre_forward to fix the CUDA device context.
    """
    for module in model.modules():
        hook = getattr(module, "_hf_hook", None)
        if hook is None or not hasattr(hook, "execution_device"):
            continue
        exec_dev = hook.execution_device
        if exec_dev is None or str(exec_dev) == "cpu":
            continue

        original_pre_forward = hook.pre_forward

        def make_patched(orig, dev):
            def patched_pre_forward(module, *args, **kwargs):
                torch.cuda.set_device(dev)
                return orig(module, *args, **kwargs)
            return patched_pre_forward

        hook.pre_forward = make_patched(original_pre_forward, exec_dev)


class StudentModel:
    def __init__(self, config: PipelineConfig, student_config: StudentConfig, paths: Paths):
        self.config = config
        self.student_config = student_config
        self.paths = paths
        self.model_path = student_config.model_path
        self.model_name = Path(student_config.model_path).name
        self.context_len = student_config.context_len
        self.model = None
        self.tokenizer = None
        self.byte_vocab = None
        self.optimizer = None
        self.lr_scheduler = None
        self.logger = None
        self.rank = 0
        self.world_size = 1

    def _log(self, msg):
        if self.rank == 0:
            print(msg)

    def _barrier(self):
        if self.world_size > 1:
            dist.barrier()

    def train(self, train_target, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        c = self.config

        self._log(f"  Preparing tokenizer and byte vocab index for {self.model_name}...")
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        if self.student_config.chat_template:
            self.tokenizer.chat_template = self.student_config.chat_template
        self.byte_vocab = ByteVocabIndex(self.tokenizer, device=f"cuda:{rank}")

        save_roles = set(c.save_roles)
        self._log("  Preprocessing training data...")
        train_samples = read_jsonl(c.dataset_path)
        train_convos = preprocess_samples(train_samples[rank::world_size], self.tokenizer, self.context_len, save_roles)
        self._log("  Preprocessing validation data...")
        val_samples = read_jsonl(c.validation_dataset_path)
        val_convos = preprocess_samples(val_samples, self.tokenizer, self.context_len, save_roles)
        val_convos.sort(key=lambda cv: cv.length, reverse=True)

        resume_state = None
        checkpoint_dir = None
        if self.student_config.resume_from:
            checkpoint_dir = resolve_checkpoint_dir(
                self.student_config.resume_from, self.paths.student_states, self.model_name
            )
            self.model_path = checkpoint_dir
            self._log(f"  Resuming from checkpoint: {checkpoint_dir}")

        if self.rank == 0:
            cleanup_incomplete_checkpoints(self.paths.student_states)
            cleanup_incomplete_checkpoints(self.paths.student_trained)
        self._barrier()

        self._log(f"  Loading model: {self.model_name}...")
        self._load_model(rank)
        self.model = wrap_distributed(self.model, self.config.training_strategy, self.config.training_precision, rank, world_size)

        if c.torch_compile:
            self._log("  Compiling model with torch.compile...")
            logging.getLogger("torch._dynamo").setLevel(logging.ERROR)
            torch._dynamo.config.allow_unspec_int_on_nn_module = True
            # Per-layer compilation avoids layer_idx recompilation (5-8x faster compile)
            base_model = self.model.module if hasattr(self.model, "module") else self.model
            for layer in get_transformer_layers(base_model):
                layer.compile(backend=c.torch_compile_backend, mode=c.torch_compile_mode)

        dataset_steps = len(train_convos)
        total_steps = dataset_steps * c.num_epochs
        eff_batch_size = c.batch_size * c.grad_accum_batches
        grad_accum_steps = total_steps // eff_batch_size
        warmup_steps = math.ceil(c.num_warmup_steps / eff_batch_size)

        self.optimizer = set_optimizer(self.model, c.lr, c.adam_betas, c.optimizer, c.adam_decay, total_steps=grad_accum_steps, training_strategy=self.config.training_strategy, warmup_steps=warmup_steps)
        if c.optimizer.lower() == "schedulefree":
            self.lr_scheduler = None
        else:
            self.lr_scheduler = set_lr_scheduler(
                self.optimizer, c.lr_scheduler, warmup_steps,
                grad_accum_steps, dataset_steps, c.lr_decay_start, c.lr, 1e-9
            )
        self._log(f"  Optimizer: {c.optimizer}, Scheduler: {c.lr_scheduler}, LR: {c.lr}")

        if checkpoint_dir:
            resume_state = load_training_state(
                checkpoint_dir, self.model, self.optimizer, self.lr_scheduler,
                self.config.training_strategy, rank, world_size
            )
            self._log(f"  Restored training state: epoch={resume_state.epoch}, num_trained={resume_state.num_trained}")

        if self.rank == 0:
            self._setup_logger(resume_state)

        data_manager = H5DataManager(self.paths.dataset, teacher_name=train_target, read_only=True)
        val_data_manager = H5DataManager(self.paths.dataset_validation, teacher_name=train_target, read_only=True)

        train_available = data_manager.get_dataset_ids()
        val_available = val_data_manager.get_dataset_ids()
        train_before = len(train_convos)
        val_before = len(val_convos)
        train_convos = [convo for convo in train_convos if convo.origin_convo_id in train_available]
        val_convos = [convo for convo in val_convos if convo.origin_convo_id in val_available]
        train_dropped = train_before - len(train_convos)
        val_dropped = val_before - len(val_convos)
        if train_dropped > 0:
            print(f"[WARN] Dropped {train_dropped}/{train_before} training convos not found in teacher dataset '{train_target}'")
        if val_dropped > 0:
            print(f"[WARN] Dropped {val_dropped}/{val_before} validation convos not found in teacher dataset '{train_target}'")

        if self.world_size > 1:
            local_count = torch.tensor([len(train_convos)], dtype=torch.long, device=f"cuda:{self.rank}")
            all_counts = [torch.zeros(1, dtype=torch.long, device=f"cuda:{self.rank}") for _ in range(self.world_size)]
            dist.all_gather(all_counts, local_count)
            max_count = max(int(t.item()) for t in all_counts)
            assert max_count > 0, "No training conversations available after filtering"
            if len(train_convos) < max_count:
                assert len(train_convos) > 0, f"Rank {self.rank} has 0 conversations after filtering, cannot pad"
                original_len = len(train_convos)
                train_convos.extend(train_convos[i % original_len] for i in range(max_count - original_len))

        final_meta = None
        interrupted = False
        try:
            final_meta = self._run_training(train_convos, val_convos, data_manager, val_data_manager, resume_state=resume_state)
        except KeyboardInterrupt:
            interrupted = True
            self._log("\n  Training interrupted by user.")
        finally:
            data_manager.close()
            val_data_manager.close()
            if self.logger:
                try:
                    wandb.finish()
                except BaseException:
                    pass

        if interrupted:
            return

        self._save_model("final")

        if self.student_config.save_final_training_state:
            self._save_training_state("final", final_meta)

        if self.rank == 0 and final_meta is not None:
            save_path = os.path.join(self.paths.student_trained, f"{self.model_name}_step_final")
            print(f"\n  Training Summary:")
            print(f"    Steps trained: {final_meta.num_trained}")
            if final_meta.best_val_loss is not None:
                print(f"    Best validation loss: {final_meta.best_val_loss:.4f}")
            print(f"    Final model saved to: {save_path}")

        self.model = None
        self.optimizer = None
        self.lr_scheduler = None
        torch.cuda.empty_cache()
        gc.collect()

    def _load_model(self, rank):
        c = self.config
        sc = self.student_config

        precision_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16, "4bit": torch.float16, "8bit": torch.float16}
        dtype = precision_map.get(c.training_precision, torch.float16)

        bnb_config = None
        if c.training_precision in ("4bit", "8bit"):
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=(c.training_precision == "4bit"),
                load_in_8bit=(c.training_precision == "8bit"),
                bnb_4bit_compute_dtype=dtype if c.training_precision == "4bit" else None,
                bnb_4bit_use_double_quant=(c.training_precision == "4bit"),
                bnb_4bit_quant_type="nf4" if c.training_precision == "4bit" else None,
            )

        strategy = c.training_strategy
        attn_impl = sc.attn_implementation

        if c.liger_kernel:
            try:
                from liger_kernel.transformers import AutoLigerKernelForCausalLM
                ModelClass = AutoLigerKernelForCausalLM
            except ImportError:
                raise ImportError("liger_kernel=true but liger-kernel not installed. Install via: pip install liger-kernel --no-deps")
            liger_kwargs = dict(cross_entropy=False, fused_linear_cross_entropy=False)
            print(f"  Using Liger Kernel (fused RoPE, RMSNorm, SwiGLU)")
        else:
            ModelClass = AutoModelForCausalLM
            liger_kwargs = {}

        if strategy == "naive_layer_split":
            if not c.multi_gpu:
                device_map = {"": f"cuda:{rank}"}
            elif c.device_map == "custom":
                device_map = _build_custom_device_map(self.model_path, self.config.num_gpu0_layers, self.config.max_memory_hf)
            else:
                device_map = c.device_map
            self.model = ModelClass.from_pretrained(
                self.model_path, device_map=device_map,
                dtype=dtype if not bnb_config else None,
                quantization_config=bnb_config, attn_implementation=attn_impl,
                max_memory=c.max_memory_hf if c.max_memory else None,
                **liger_kwargs,
            )
        else:
            self.model = ModelClass.from_pretrained(
                self.model_path, dtype=dtype if not bnb_config else None,
                quantization_config=bnb_config, attn_implementation=attn_impl,
                **liger_kwargs,
            )
            if strategy != "fsdp2":
                self.model = self.model.to(f"cuda:{rank}")

        if c.liger_kernel and strategy == "naive_layer_split" and c.multi_gpu:
            _patch_device_hooks(self.model)

        self.model.train()
        if c.grad_checkpointing:
            self.model.gradient_checkpointing_enable()

        for name, param in self.model.named_parameters():
            if any(fl in name for fl in sc.freeze_layers):
                param.requires_grad = False

    def _run_training(self, train_convos, val_convos, data_manager, val_data_manager, resume_state=None):
        c = self.config
        eff_batch_size = c.batch_size * c.grad_accum_batches

        val_batches, val_id_batches = self._construct_batches(val_convos)
        val_data_manager.read_only_mode(val_id_batches)

        if resume_state:
            num_trained = resume_state.num_trained
            next_accum = resume_state.next_accum
            next_val = resume_state.next_val
            next_save = resume_state.next_save
            next_state_save = resume_state.next_state_save
            start_epoch = resume_state.epoch
            best_val_loss = resume_state.best_val_loss
        else:
            num_trained = 0
            next_accum = eff_batch_size
            next_val = int(c.validate_every_n_epochs * len(train_convos))
            next_save = int(c.save_student_every_n_epochs * len(train_convos))
            start_epoch = 0
            best_val_loss = None
            self._validate(val_batches, val_data_manager, num_trained, pbar=None)
            self._barrier()

        if hasattr(self.optimizer, 'train'):
            self.optimizer.train()

        pbar = tqdm(total=len(train_convos) * c.num_epochs - num_trained, desc="Training", disable=(self.rank != 0), leave=False, smoothing=0.06)
        last_postfix = ""

        sc = self.student_config
        total_samples = len(train_convos) * c.num_epochs
        if sc.save_training_state_every_n_epochs:
            epoch_size = len(train_convos)
            state_save_interval = int(sc.save_training_state_every_n_epochs * epoch_size)
            if not resume_state:
                next_state_save = state_save_interval
        else:
            state_save_interval = None
            if not resume_state:
                next_state_save = None
            else:
                next_state_save = None

        epoch = start_epoch
        for epoch in range(start_epoch, c.num_epochs):
            self._reorder(train_convos, c.data_order, epoch)

            batches, id_batches = self._construct_batches(train_convos)

            if resume_state and epoch == start_epoch:
                samples_this_epoch = resume_state.num_trained - start_epoch * len(train_convos)
                batches_to_skip = samples_this_epoch // c.batch_size
                if batches_to_skip > 0:
                    batches = batches[batches_to_skip:]
                    id_batches = id_batches[batches_to_skip:]
                    self._log(f"  Skipping {batches_to_skip} batches ({samples_this_epoch} samples) in epoch {epoch}")

            data_manager.enqueue_get_batches(id_batches)

            losses = Losses(self.logger)

            for batch_convos in batches:
                self._forward_batch(batch_convos, data_manager, losses, training=True)

                is_accum_step = num_trained + len(batch_convos) < next_accum
                if is_accum_step and isinstance(self.model, DDP):
                    with self.model.no_sync():
                        losses.backward(divisor=eff_batch_size)
                elif is_accum_step and hasattr(self.model, 'set_requires_gradient_sync'):
                    self.model.set_requires_gradient_sync(False)
                    losses.backward(divisor=eff_batch_size)
                    self.model.set_requires_gradient_sync(True)
                else:
                    losses.backward(divisor=eff_batch_size)

                num_trained += len(batch_convos)
                pbar.update(len(batch_convos))

                if num_trained >= next_accum:
                    if c.max_grad_norm > 0:
                        grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), c.max_grad_norm)
                    else:
                        grad_norm = torch.tensor(0.0)
                    self.optimizer.step()
                    self.optimizer.zero_grad(set_to_none=True)
                    if self.lr_scheduler:
                        self.lr_scheduler.step()

                    if self.rank == 0:
                        avg_loss = losses.loss_dict.get("custom loss", torch.tensor(0.0)).item() / max(losses.num_steps_accumulated, 1)
                        if self.logger:
                            self.logger.log({"Learning Rate": self.optimizer.param_groups[0]["lr"], "grad_norm": grad_norm.item()}, step=num_trained)
                        losses.log(num_trained)
                        pbar.set_postfix_str(f"loss={avg_loss:.4f} lr={self.optimizer.param_groups[0]['lr']:.2e} gnorm={grad_norm:.2f}")
                        last_postfix = pbar.postfix
                    losses.empty()
                    next_accum += eff_batch_size

                    next_state_save = self._maybe_save_training_state_checkpoint(
                        num_trained, epoch, next_accum, next_val, next_save, next_state_save, state_save_interval,
                        best_val_loss, total_samples, pbar, last_postfix,
                    )

                next_save = self._maybe_save_model_checkpoint(
                    num_trained, total_samples, next_save, train_convos, pbar, last_postfix,
                )

                next_val, best_val_loss = self._maybe_validate_and_save(
                    num_trained, epoch, next_accum, next_val, next_save, next_state_save,
                    best_val_loss, train_convos, val_batches, val_data_manager, pbar, last_postfix,
                )

        pbar.close()
        return self._make_state(num_trained, epoch, next_accum, next_val, next_save, next_state_save, best_val_loss)

    def _make_state(self, num_trained, epoch, next_accum, next_val, next_save, next_state_save, best_val_loss):
        return TrainingState(
            num_trained=num_trained, epoch=epoch, next_accum=next_accum,
            next_val=next_val, next_save=next_save, next_state_save=next_state_save,
            wandb_run_id=self.logger.id if self.logger else None, best_val_loss=best_val_loss,
        )

    def _maybe_save_training_state_checkpoint(self, num_trained, epoch, next_accum, next_val, next_save,
                                              next_state_save, state_save_interval, best_val_loss,
                                              total_samples, pbar, last_postfix):
        if next_state_save is None or num_trained < next_state_save:
            return next_state_save
        next_state_save += state_save_interval
        is_final = self.student_config.save_final_training_state and num_trained >= total_samples
        if not is_final:
            pbar.set_postfix_str(f"Saving training state at step {num_trained}...")
            metadata = self._make_state(num_trained, epoch, next_accum, next_val, next_save, next_state_save, best_val_loss)
            self._save_training_state(num_trained, metadata)
            if self.rank == 0 and self.config.keep_last_n_checkpoints:
                pbar.set_postfix_str("Deleting old checkpoints...")
                rotate_checkpoints(self.paths.student_states, f"{self.model_name}_training_state_step_", self.config.keep_last_n_checkpoints)
            pbar.set_postfix_str(last_postfix)
        return next_state_save

    def _maybe_save_model_checkpoint(self, num_trained, total_samples, next_save, train_convos,
                                     pbar, last_postfix):
        if num_trained < next_save:
            return next_save
        is_final = num_trained >= total_samples
        if not is_final:
            pbar.set_postfix_str(f"Saving checkpoint at step {num_trained}...")
            self._save_model(num_trained)
            if self.rank == 0 and self.config.keep_last_n_checkpoints:
                rotate_checkpoints(self.paths.student_trained, f"{self.model_name}_step_", self.config.keep_last_n_checkpoints)
            pbar.set_postfix_str(last_postfix)
        return next_save + int(self.config.save_student_every_n_epochs * len(train_convos))

    def _maybe_validate_and_save(self, num_trained, epoch, next_accum, next_val, next_save, next_state_save,
                                 best_val_loss, train_convos, val_batches, val_data_manager, pbar, last_postfix):
        if num_trained < next_val:
            return next_val, best_val_loss
        c = self.config
        pbar.set_postfix_str("Validating...")
        val_loss = self._validate(val_batches, val_data_manager, num_trained, pbar=pbar, best_val_loss=best_val_loss)
        save_best = False
        if self.rank == 0 and val_loss is not None:
            if best_val_loss is None or val_loss < best_val_loss:
                best_val_loss = val_loss
                save_best = True
        if self.world_size > 1:
            save_best_t = torch.tensor([1 if save_best else 0], device=f"cuda:{self.rank}")
            dist.broadcast(save_best_t, src=0)
            save_best = save_best_t.item() == 1
        if save_best:
            if c.save_best_model:
                pbar.set_postfix_str("Saving best model...")
                self._save_model("best")
            if c.save_best_state:
                pbar.set_postfix_str("Saving best training state...")
                metadata = self._make_state(num_trained, epoch, next_accum, next_val, next_save, next_state_save, best_val_loss)
                self._save_training_state("best", metadata)
        pbar.set_postfix_str(last_postfix)
        self._barrier()
        next_val += int(c.validate_every_n_epochs * len(train_convos))
        return next_val, best_val_loss

    def _forward_batch(self, batch_convos, data_manager, losses, training=False):
        with data_manager.read_next_batch() as batch:
            teacher_batch = torch.from_numpy(batch.data).to(f"cuda:{self.rank}", non_blocking=True).float()

            max_len = max(cv.length for cv in batch_convos)
            tokens_np = np.array([cv.tokens[:max_len] for cv in batch_convos])
            tokens_t = torch.from_numpy(tokens_np).to(f"cuda:{self.rank}", non_blocking=True)

            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                logits_all = self.model(tokens_t, use_cache=False).logits

            if logits_all.device != tokens_t.device:
                logits_all = logits_all.to(tokens_t.device)
            torch.cuda.set_device(self.rank)

            for i, convo in enumerate(batch_convos):
                td = teacher_batch[i][:batch.padding[i]] if batch.padding[i] > 0 else teacher_batch[i]
                loss_dict = self._compute_sample_loss(logits_all[i].float(), tokens_t[i], convo, td, training)
                if loss_dict is not None:
                    losses.add_losses(loss_dict)

            del logits_all, teacher_batch

    def _compute_sample_loss(self, logits, token_ids, convo, teacher_dists, training):
        actual_bytes_t = torch.from_numpy(convo.actual_bytes).to(logits.device)

        T = self.student_config.training_temperature
        if T != 1.0:
            logits = logits / T

        loss_dict = None
        if training:
            loss_dict = _try_fused_train(
                logits, token_ids, convo, teacher_dists, actual_bytes_t,
                self.byte_vocab, self.config.alpha, self.config.loss_type,
                entropy_weighting=self.config.entropy_weighting,
            )

        if loss_dict is None:
            student_byte_dists = self.byte_vocab.marginalize_content(
                logits, token_ids, convo.content_byte_ranges,
                length=convo.length, T_CHUNK=self.config.marg_chunk_training, training=training,
            ).float()

            n = min(student_byte_dists.shape[0], teacher_dists.shape[0])
            if n == 0:
                if training:
                    return {"train_loss": torch.tensor(0.0, device=logits.device, requires_grad=True)}
                return None
            student_byte_dists = student_byte_dists[:n]
            teacher_dists = teacher_dists[:n]
            actual_bytes_t = actual_bytes_t[:n]

            entropy_weights = None
            if self.config.entropy_weighting:
                eps = 1e-8
                t = teacher_dists + eps
                H = -(t * t.log()).sum(-1)
                entropy_weights = H / 5.545177  # log(256)

            loss_dict = calculate_divergence(student_byte_dists, teacher_dists, actual_bytes_t, self.config.alpha, self.config.loss_type, entropy_weights=entropy_weights)

        if training and torch.isnan(loss_dict["train_loss"]):
            print(f"[WARN] NaN loss at step, substituting zero loss")
            zero = torch.tensor(0.0, device=logits.device, requires_grad=True)
            loss_dict = {k: (zero if k == "train_loss" else torch.tensor(0.0, device=logits.device)) for k in loss_dict}

        if T != 1.0:
            t_sq = T * T
            for key in ("train_loss", "custom loss"):
                if key in loss_dict:
                    loss_dict[key] = loss_dict[key] * t_sq

        return loss_dict

    def _validate(self, val_batches, val_data_manager, step, pbar=None, best_val_loss=None):
        self.model.eval()
        if hasattr(self.optimizer, 'eval'):
            self.optimizer.eval()
        losses = Losses(self.logger, validation=True)

        with torch.no_grad():
            iterator = tqdm(val_batches, desc="  Validating", leave=False, smoothing=0.06, disable=(self.rank != 0))
            for batch_convos in iterator:
                self._forward_batch(batch_convos, val_data_manager, losses)

        avg_loss = None
        tracked_metric = None
        if self.rank == 0:
            n = max(losses.num_steps_accumulated, 1)
            avg_loss = losses.loss_dict.get("custom loss", torch.tensor(0.0)).item() / n
            avg_ce = losses.loss_dict.get("CE loss", torch.tensor(0.0)).item() / n
            avg_kl = losses.loss_dict.get("kl_div", torch.tensor(0.0)).item() / n
            msg = f"  Validation - loss: {avg_loss:.4f}, CE: {avg_ce:.4f}, KL: {avg_kl:.4f}"

            metric_key = {"train_loss": "custom loss", "ce": "CE loss", "kl": "kl_div"}.get(self.config.best_metric, "custom loss")
            tracked_metric = losses.loss_dict.get(metric_key, torch.tensor(0.0)).item() / n
            if best_val_loss is None or tracked_metric < best_val_loss:
                msg += " <- new best achieved!"
            tqdm.write(msg) if pbar else print(msg)
            losses.log(step)

        losses.empty()
        self.model.train()
        if hasattr(self.optimizer, 'train'):
            self.optimizer.train()
        return tracked_metric

    def _construct_batches(self, convos):
        bs = self.config.batch_size
        convo_batches = [convos[i:i+bs] for i in range(0, len(convos), bs)]
        id_batches = [[c.origin_convo_id for c in batch] for batch in convo_batches]
        return convo_batches, id_batches

    def _reorder(self, convos, order, epoch=0):
        if order in ("shuffle", "random"):
            rng = np.random.RandomState(self.config.seed + epoch)
            rng.shuffle(convos)
        elif order == "sorted":
            convos.sort(key=lambda c: c.length, reverse=True)

    def _save_model(self, step):
        folder = os.path.join(self.paths.student_trained, f"{self.model_name}_step_{step}")
        save_model(self.model, self.tokenizer, folder, self.config.training_strategy, self.rank, self.world_size)

    def _save_training_state(self, step, metadata):
        folder = os.path.join(self.paths.student_states, f"{self.model_name}_training_state_step_{step}")
        save_training_state(self.model, self.tokenizer, self.optimizer, self.lr_scheduler, folder, self.config.training_strategy, self.rank, self.world_size, metadata)

    def _setup_logger(self, resume_state=None):
        c = self.config
        name = f"{c.wandb_comment + ' ' if c.wandb_comment else ''}{self.model_name} lr({c.lr}) ({time.strftime('%d.%m.%Y / %H:%M:%S')})"
        project = c.wandb_project or "LLM Distillation"
        config_dict = {
            "model": self.model_path, "context_len": self.context_len,
            "lr": c.lr, "epochs": c.num_epochs, "batch_size": c.batch_size,
            "grad_accum": c.grad_accum_batches, "precision": c.training_precision,
            "optimizer": c.optimizer, "scheduler": c.lr_scheduler,
            "alpha": c.alpha,
        }
        try:
            wandb_kwargs = dict(
                project=project, name=name, group=self.model_name,
                dir=self.paths.cache, config=config_dict,
                settings=wandb.Settings(quiet=True),
            )
            if resume_state and resume_state.wandb_run_id:
                wandb_kwargs["id"] = resume_state.wandb_run_id
                wandb_kwargs["resume"] = "allow"
            else:
                wandb_kwargs["reinit"] = "finish_previous"
            self.logger = wandb.init(**wandb_kwargs)
        except Exception as e:
            print(f"Warning: wandb init failed ({e}), continuing without wandb logging")
            self.logger = None

