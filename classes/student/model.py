from utils.finetuning_utils import set_optimizer, set_lr_scheduler, calculate_divergence
from peft import get_peft_config, get_peft_model, LoraConfig, TaskType, PeftModel
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
from transformers import AutoModelForCausalLM, AutoTokenizer
from classes.data_classes import ConvoTokenized
from classes.data_manager import H5DataManager
from multiprocessing import shared_memory
from classes.base_model import BaseModel
from classes.paths import Paths
from typing import Optional
from copy import deepcopy
from tqdm import tqdm
import torch.nn.functional as F
import peft.utils as peft_utils
import numpy as np
import torch
import wandb
import math
import time
import os


os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['WANDB_SILENT'] = 'true'

class StudentModel(BaseModel):
    def __init__(self, model_path: str, paths: Paths, add_bos: bool, prompt_format: dict, batch_size: int):
        super().__init__(model_path, student=True)

        self.model: Optional[AutoModelForCausalLM|PeftModel] = None
        self.adapter = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.add_bos = add_bos
        self.prompt_format = prompt_format
        self.batch_size = batch_size
        self.paths: Paths = paths

        self.optimizer_name = ""
        self.optimizer = None
        self.lr_scheduler_name = ""
        self.lr_scheduler = None
        self.lr = 0.0
        self.decay_start = 0.0 # wsd only
        self.final_lr = 5e-7 # wsd only

        self.num_epochs = 0
        self.num_training_steps = 0
        self.num_warmup_steps = 0
        self.num_grad_accum_batches = 0
        self.validation_every_steps = 0
        self.save_every_steps = 0

        # state and progress tracking
        self.next_save_step = 0
        self.num_trained_steps = 0
        self.next_val_step = 0
        self.next_accum_step = 0
        self.next_merge_step = 0
        self.state_path = ""
        self.distr_device = ""
        self.saved_state = False
        self.val_batch_order_ids = []

        # dora
        self.use_dora = False
        self.lora_alpha = 0.0
        self.lora_rank = 0
        self.target_modules = []
        self.perma_merge_weight = 0.0
        self.perma_merge_every_batches = 0
        self.peft_config = None
        
        # misc
        self.freeze_layers = []
        self.training_precision_name = ""
        self.custom_reduction = False
        self.logger = None
        self.data_order = ""
        self.save_final_state = False
        self.grad_checkpointing = False
        self.multi_gpu = False
        


    def _set_postfix(self, postfix: str):
        self.progress_bar.set_postfix_str(postfix)

    def _release_postfix(self):
        self.progress_bar.set_postfix_str("")

    def _get_content_indices_tensor(self, content_ranges) -> torch.Tensor:
        content_indices = []
        for start, end in content_ranges:
            content_indices.append(np.arange(start, end))
        return torch.as_tensor(np.concatenate(content_indices), dtype=torch.long).to(self.device, non_blocking=True)

    def _load_model(self, model_path: str = None):
        self._set_postfix("Loading model...")

        precision_dict = {
            "fp16": torch.float16,
            "4bit": torch.float16,
            "8bit": torch.float16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16
        }
        
        train_precision = precision_dict.get(self.training_precision_name, torch.float16)

        path = model_path if model_path is not None else self.model_path

        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="balanced" if self.multi_gpu else self.device, #"balanced_low_0"
            torch_dtype=train_precision,
            load_in_4bit=self.training_precision_name == "4bit",
            load_in_8bit = self.training_precision_name == "8bit",
            attn_implementation="flash_attention_2" if not self.use_dora else None
        )

        if self.use_dora:
            self.peft_config = LoraConfig(use_dora=True, task_type=TaskType.CAUSAL_LM, lora_alpha=self.lora_alpha, r=self.lora_rank)

            self.model = get_peft_model(self.model, self.peft_config)

            self.model.set_adapter("default")
            self.model.print_trainable_parameters()

        else:
            self.model.train()
            if self.grad_checkpointing:
                self.model.gradient_checkpointing_enable()

            if self.freeze_layers:
                for name, param in self.model.named_parameters():
                    for freeze_layer in self.freeze_layers:
                        if freeze_layer in name:
                            param.requires_grad = False

        self._release_postfix()

    def _unload_model(self):
        self.model = None

    def _load_optimizer(self):
        self._set_postfix("Loading optimizer...")

        self.optimizer = set_optimizer(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.9, 0.999),
            optimizer_name=self.optimizer_name,
            weight_decay=2e-5
        )

        self._release_postfix()

    def _unload_optimizer(self):
        self.optimizer = None
    
    def _sort_val_dataset_by_len(self):
        self.validation_dataset.sort(key=lambda convo: convo.length, reverse=True)
        self.validation_dataset_sorted = True
    
    def _reorder_dataset(self):
        if self.dataset_sorted:
            return
        if self.data_order == "shuffle":
            np.random.shuffle(self.dataset)
        elif self.data_order == "sorted":
            self.dataset.sort(key=lambda convo: convo.length, reverse=True)
            self.dataset_sorted = True
        elif self.data_order == "native":
            self.dataset_sorted = True

    def _save_state(self):
        self._set_postfix("Saving state...")

        if self.num_trained_steps < self.num_training_steps:
            self.paths.empty_student_states()

        torch.save(self.model.state_dict(), os.path.join(self.paths.student_states, "model_state.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.paths.student_states, "optimizer_state.pt"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(self.paths.student_states, "scheduler_state.pt"))

        self._release_postfix()

    def _save_model(self, step: int|str):
        self._set_postfix(f"Saving model at step {step}...")

        folder_name = f"{self.model_name}_step_{step}"

        if self.use_dora:
            self.model.save_pretrained(os.path.join(self.paths.student_trained, folder_name, "adapter"))
        else:
            self.model.save_pretrained(os.path.join(self.paths.student_trained, folder_name))

        self.tokenizer.save_pretrained(os.path.join(self.paths.student_trained, folder_name))

        self._release_postfix()

    def _load_state(self):
        self._set_postfix("Loading state...")

        model_state = torch.load(os.path.join(self.paths.student_states, "model_state.pt"), map_location=self.device)
        optimizer_state = torch.load(os.path.join(self.paths.student_states, "optimizer_state.pt"), map_location=self.device)
        scheduler_state = torch.load(os.path.join(self.paths.student_states, "scheduler_state.pt"), map_location=self.device)

        self._load_model()
        self.model.load_state_dict(model_state, assign=True)

        self._load_optimizer()
        self.optimizer.load_state_dict(optimizer_state)

        self.lr_scheduler = set_lr_scheduler(self.optimizer, self.lr_scheduler_name, self.num_warmup_steps, self.num_training_steps, self.dataset_len, self.decay_start, self.lr, self.final_lr) 
        self.lr_scheduler.load_state_dict(scheduler_state)

        self._release_postfix()

    def _gradual_lora_merge(self):
        self._set_postfix("Gradually Merging Lora...")

        adapter = peft_utils.save_and_load.get_peft_model_state_dict(self.model)

        diff = deepcopy(adapter)

        for key, value in diff.items():
            diff[key] = value * self.perma_merge_weight

        self._release_postfix()

    def _construct_batches(self, dataset_chunk: list[ConvoTokenized]) -> tuple[list[list[ConvoTokenized]], list[list[int]]]:
        num_batches = math.ceil(len(dataset_chunk) / self.batch_size)
        convo_batches = []
        id_batches = []

        for i in range(num_batches):
            batch = dataset_chunk[i*self.batch_size:(i+1)*self.batch_size]
            convo_batches.append(batch)
            id_batches.append([convo.origin_convo_id for convo in batch])

        return convo_batches, id_batches

    def train_chunk(self, data_manager: H5DataManager, validation_data_manager: H5DataManager, full_collect):
        trainable_ids = data_manager.get_dataset_ids()

        if not self.validation_dataset_sorted:
            self._sort_val_dataset_by_len()
            self.validation_dataset_batched, self.val_batch_order_ids = self._construct_batches(self.validation_dataset)
            validation_data_manager.read_only_mode(self.val_batch_order_ids)

        if self.logger is None:
            comment = f"{self.model_name}_lr_{self.lr}_{time.strftime('%d-%m-%Y_%H-%M-%S')}"
            self.logger = wandb.init(project="student_training", name=comment, config=self.__dict__, group=self.model_name, reinit=True)

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.progress_bar = tqdm(total=self.num_training_steps, initial=self.num_trained_steps, desc="Training", smoothing=0.06, leave=False)

        if self.saved_state:
            self._load_state()
            self.paths.empty_student_states()
            self.saved_state = False
        else:
            self._load_model()
            self._load_optimizer()
            self.lr_scheduler = set_lr_scheduler(self.optimizer, self.lr_scheduler_name, self.num_warmup_steps, self.num_training_steps, self.dataset_len, self.decay_start, self.lr, self.final_lr) 

        if self.num_trained_steps == 0:
            self._validate_distillation(validation_data_manager)


        if full_collect:
            data_manager_left_shuffles = self.num_epochs - 1
            data_list = []
            self._reorder_dataset()
            dataset_chunk = self.dataset if full_collect else [convo for convo in self.dataset if convo.origin_convo_id in trainable_ids]
            batched_chunk_convos, id_batches = self._construct_batches(dataset_chunk)
            data_list.append(batched_chunk_convos)
            data_manager.enqueue_get_batches(id_batches)
                
            for i in range(self.num_epochs):
                if data_manager_left_shuffles > 0:
                    self._reorder_dataset()
                    dataset_chunk = self.dataset if full_collect else [convo for convo in self.dataset if convo.origin_convo_id in trainable_ids]
                    batched_chunk_convos, id_batches = self._construct_batches(dataset_chunk)
                    data_list.append(batched_chunk_convos)
                    data_manager.enqueue_get_batches(id_batches)
                    data_manager_left_shuffles -= 1

                self._run_distillation_cycle(data_list.pop(0), data_manager, validation_data_manager)
        else:
            self._reorder_dataset()
            dataset_chunk = self.dataset if full_collect else [convo for convo in self.dataset if convo.origin_convo_id in trainable_ids]
            batched_chunk_convos, id_batches = self._construct_batches(dataset_chunk)
            data_manager.enqueue_get_batches(id_batches)
            self._run_distillation_cycle(batched_chunk_convos, data_manager, validation_data_manager)

        if not full_collect:
            self._save_state()
            self.saved_state = True

        if self.num_trained_steps >= self.num_training_steps:
            self._save_model("final")
            if self.save_final_state:
                self._save_state()
                self.saved_state = True

        self.progress_bar.close()
        self._unload_model()
        self._unload_optimizer()
        torch.cuda.empty_cache()
        self.lr_scheduler = None

    def close(self):
        if self.logger is not None:
            self.logger.finish()

    # main training loop
    def _run_distillation_cycle(self, batched_chunk_convos, data_manager: H5DataManager, validation_data_manager: H5DataManager):
        print("Running distillation cycle")
        for batch_convos in batched_chunk_convos:
            num_steps = len(batch_convos)
            device = self.distr_device if self.distr_device else self.device

            sdh_mem_name, batch_shape, batch_dtype, batch_indices, batch_padding = data_manager.read_next_batch()
            shd_mem = shared_memory.SharedMemory(name=sdh_mem_name)
            teacher_batch_raw = torch.from_numpy(np.ndarray(batch_shape, dtype=batch_dtype, buffer=shd_mem.buf)).to(device, non_blocking=True)

            max_non_padded_len = max(convo.length for convo in batch_convos)
            batch_tokenized = np.array([convo.tokenized[:max_non_padded_len] for convo in batch_convos])
            batch_tokenized_tensor = torch.from_numpy(batch_tokenized).to(self.device, non_blocking=True)

            batch_combined_loss = torch.tensor(0.0).to(device, non_blocking=True)
            batch_cross_entropy_loss = torch.tensor(0.0).to(device, non_blocking=True)
            batch_custom_loss = torch.tensor(0.0).to(device, non_blocking=True)
            batch_kl_div = torch.tensor(0.0).to(device, non_blocking=True)
            batch_variance = torch.tensor(0.0).to(device, non_blocking=True)
            batch_certainty_loss = torch.tensor(0.0).to(device, non_blocking=True)
            print("Done initializing tensors")

            batch_logits = self.model(batch_tokenized_tensor).logits[:, :, :self.crop_to_size].float()
            print("Done forward pass")

            if not self.distr_device:
                self.distr_device = batch_logits.device

            for i, convo in enumerate(batch_convos):
                indices = torch.from_numpy(batch_indices[i]).to(device, non_blocking=True)
                content_indices = self._get_content_indices_tensor(convo.content_ranges)

                convo_content_logits = torch.index_select(batch_logits[i], 0, content_indices) / self.temperature
                convo_teacher_logits = teacher_batch_raw[i]
                convo_content_tokens = torch.index_select(batch_tokenized_tensor[i], 0, content_indices)

                combined_loss, cross_entropy_loss, custom_loss, kl_div, variance, certainty_loss = calculate_divergence(convo_content_logits, convo_teacher_logits[:batch_padding[i]], indices, convo_content_tokens, custom=self.custom_reduction)
                print("Done calculate_divergence")
                divisor = num_steps * self.num_grad_accum_batches

                batch_combined_loss += combined_loss / divisor
                batch_cross_entropy_loss += cross_entropy_loss / divisor
                batch_custom_loss += custom_loss / divisor
                batch_kl_div += kl_div / divisor
                batch_variance += variance / divisor
                batch_certainty_loss += certainty_loss / divisor

                self.lr_scheduler.step()
                print("Done lr_scheduler step")

            self.logger.log({"Loss/combined (custom + variance + certainty)": batch_combined_loss}, step=self.num_trained_steps)
            self.logger.log({"Loss/cross entropy": batch_cross_entropy_loss}, step=self.num_trained_steps)
            self.logger.log({"Loss/custom loss": batch_custom_loss}, step=self.num_trained_steps)
            self.logger.log({"Loss/KL divergence": batch_kl_div}, step=self.num_trained_steps)
            self.logger.log({"Loss/variance": batch_variance}, step=self.num_trained_steps)
            self.logger.log({"Loss/certainty loss": batch_certainty_loss}, step=self.num_trained_steps)

            self.logger.log({"Learning rate": self.lr_scheduler.get_last_lr()[0]}, step=self.num_trained_steps)
            self.progress_bar.update(num_steps)
            print("Done logging")
            
            batch_combined_loss.backward()
            print("Done backward pass")

            self.num_trained_steps += num_steps

            if self.num_trained_steps >= self.next_accum_step:
                self.optimizer.step()
                print("Done optimizer step")
                self.optimizer.zero_grad()
                print("Done optimizer zero_grad")
                self.next_accum_step += self.num_grad_accum_batches * self.batch_size
                
            if (self.num_trained_steps >= self.next_save_step) and not (self.num_trained_steps >= self.num_training_steps):
                self._save_model(self.num_trained_steps)
                self.next_save_step += self.save_every_steps

            if self.num_trained_steps >= self.next_val_step:
                self._validate_distillation(validation_data_manager)

    def _validate_distillation(self, validation_data_manager: H5DataManager):
        self.model.eval()
        device = self.distr_device if self.distr_device else self.device
        pbar = tqdm(total=self.validation_dataset_len, desc="Validating", leave=False, smoothing=0.06)

        with torch.no_grad():
            batch_combined_loss = torch.tensor(0.0).to(device, non_blocking=True)
            batch_cross_entropy_loss = torch.tensor(0.0).to(device, non_blocking=True)
            batch_custom_loss = torch.tensor(0.0).to(device, non_blocking=True)
            batch_kl_div = torch.tensor(0.0).to(device, non_blocking=True)
            batch_variance = torch.tensor(0.0).to(device, non_blocking=True)
            batch_certainty_loss = torch.tensor(0.0).to(device, non_blocking=True)

            for val_convo_batch in self.validation_dataset_batched:
                sdh_mem_name, batch_shape, batch_dtype, val_batch_indices, batch_padding = validation_data_manager.read_next_batch()
                shd_mem = shared_memory.SharedMemory(name=sdh_mem_name)
                teacher_batch_raw = torch.from_numpy(np.ndarray(batch_shape, dtype=batch_dtype, buffer=shd_mem.buf)).to(device, non_blocking=True)

                val_max_non_padded_len = max(convo.length for convo in val_convo_batch)
                val_batch_tokenized = np.array([convo.tokenized[:val_max_non_padded_len] for convo in val_convo_batch])
                val_batch_tokenized_tensor = torch.from_numpy(val_batch_tokenized).to(self.device, non_blocking=True)

                val_batch_logits = self.model(val_batch_tokenized_tensor).logits[:, :, :self.crop_to_size].float()

                for i, val_convo in enumerate(val_convo_batch):
                    val_indices = torch.from_numpy(val_batch_indices[i]).to(device, non_blocking=True)
                    val_content_indices = self._get_content_indices_tensor(val_convo.content_ranges)

                    val_convo_content_logits = torch.index_select(val_batch_logits[i], 0, val_content_indices) / self.temperature
                    val_convo_teacher_logits = teacher_batch_raw[i]
                    val_convo_content_tokens = torch.index_select(val_batch_tokenized_tensor[i], 0, val_content_indices)
                    
                    combined_loss, cross_entropy_loss, custom_loss, kl_div, variance, certainty_loss = calculate_divergence(val_convo_content_logits, val_convo_teacher_logits[:batch_padding[i]], val_indices, val_convo_content_tokens, custom=self.custom_reduction)
                    
                    batch_combined_loss += combined_loss
                    batch_cross_entropy_loss += cross_entropy_loss
                    batch_custom_loss += custom_loss
                    batch_kl_div += kl_div
                    batch_variance += variance
                    batch_certainty_loss += certainty_loss

                pbar.update(len(val_convo_batch))

        divisor = len(self.validation_dataset)
        self.logger.log({"Val Loss/combined (custom + variance + certainty)": batch_combined_loss/divisor}, step=self.num_trained_steps)
        self.logger.log({"Val Loss/cross entropy": batch_cross_entropy_loss/divisor}, step=self.num_trained_steps)
        self.logger.log({"Val Loss/custom loss": batch_custom_loss/divisor}, step=self.num_trained_steps)
        self.logger.log({"Val Loss/KL divergence": batch_kl_div/divisor}, step=self.num_trained_steps)
        self.logger.log({"Val Loss/variance": batch_variance/divisor}, step=self.num_trained_steps)
        self.logger.log({"Val Loss/certainty loss": batch_certainty_loss/divisor}, step=self.num_trained_steps)

        pbar.close()
        self.model.train()
        self.next_val_step += self.validation_every_steps













    def get_outliers(self, data_manager: H5DataManager):

        def _load_model_exl() -> tuple[ExLlamaV2, ExLlamaV2Cache]:
            num_gpus = torch.cuda.device_count()
            reserve_vram_kb = [int(128 * 1024**2)]*num_gpus

            config = ExLlamaV2Config()
            config.model_dir = self.model_path
            config.prepare()
            config.max_seq_len = self.context_len
            config.max_batch_size = self.batch_size
            config.max_input_len = self.seq_chunk_len
            config.max_attention_size = self.context_len ** 2

            cache = ExLlamaV2Cache(config, reserve_vram_kb)
            model = ExLlamaV2(config, cache)

            return model, cache
        
        def _unload_model_exl(model: ExLlamaV2, cache: ExLlamaV2Cache):
            model.unload()

            del model
            del cache
            torch.cuda.empty_cache()


        trainable_ids = data_manager.get_dataset_ids()

        self.dataset.sort(key=lambda convo: convo.length, reverse=True)
        dataset_chunk = [convo for convo in self.dataset if convo.origin_convo_id in trainable_ids]
        batched_chunk_convos, id_batches = self._construct_batches(dataset_chunk)
        data_manager.enqueue_get_batches(id_batches)

        model, cache = _load_model_exl()

        pbar = tqdm(total=self.dataset_len, desc="Getting outliers", leave=False, smoothing=0.06)

        for batch_convos in batched_chunk_convos:
            device = self.distr_device if self.distr_device else self.device

            sdh_mem_name, batch_shape, batch_dtype, batch_indices, batch_padding = data_manager.read_next_batch()
            shd_mem = shared_memory.SharedMemory(name=sdh_mem_name)
            teacher_batch_raw = torch.from_numpy(np.ndarray(batch_shape, dtype=batch_dtype, buffer=shd_mem.buf)).to(device, non_blocking=True)

            max_non_padded_len = max(convo.length for convo in batch_convos)
            batch_tokenized = np.array([convo.tokenized[:max_non_padded_len] for convo in batch_convos])
            batch_tokenized_tensor = torch.from_numpy(batch_tokenized).to(self.device, non_blocking=True)

            batch_logits = model.forward(batch_tokenized_tensor, cache=cache).contiguous()[:, :, :self.crop_to_size].float()

            if not self.distr_device:
                self.distr_device = batch_logits.device

            for i, convo in enumerate(batch_convos):
                indices = torch.from_numpy(batch_indices[i]).to(device, non_blocking=True)
                content_indices = self._get_content_indices_tensor(convo.content_ranges)

                teacher_padding = batch_padding[i]

                convo_content_logits = torch.index_select(batch_logits[i], 0, content_indices) / self.temperature
                convo_teacher_logits = teacher_batch_raw[i][:teacher_padding]

                min_len = min(convo_content_logits.size(0), convo_teacher_logits.size(0), indices.size(0) if indices is not None else convo_content_logits.size(0))
                
                student_logprobs = F.log_softmax(convo_content_logits, dim=-1)[:min_len]
                student_gathered = torch.gather(student_logprobs, dim=-1, index=indices[:min_len])
                teacher_logprobs = F.log_softmax(convo_teacher_logits[:min_len], dim=-1)
                
                kl_div = F.kl_div(student_gathered, teacher_logprobs, reduction='none', log_target=True).sum(dim=-1)

                pbar.update(1)


        