from utils.finetuning_utils import set_optimizer, set_lr_scheduler, calculate_divergence
from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP
from transformers import AutoModelForCausalLM, AutoTokenizer
from classes.data_classes import ConvoTokenized
from classes.data_manager import H5DataManager
from multiprocessing import shared_memory
from classes.base_model import BaseModel
from classes.paths import Paths
from typing import Optional
from tqdm import tqdm
import numpy as np
import torch
import wandb
import math
import time
import os

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
os.environ['WANDB_SILENT'] = 'true'
os.environ['ACCELERATE_USE_FSDP'] = '1'

class StudentModel(BaseModel):
    def __init__(self, model_path: str, paths: Paths, add_bos: bool, prompt_format: dict, batch_size: int):
        super().__init__(model_path, student=True)

        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.add_bos = add_bos
        self.prompt_format = prompt_format
        self.batch_size = batch_size
        self.paths: Paths = paths

        self.optimizer_name = ""
        self.optimizer = None
        self.lr_scheduler_name = ""
        self.lr_scheduler = None
        self.lr = 0
        self.decay_start = 0 # wsd only
        self.final_lr = 5e-7 # wsd only

        self.num_epochs = 0
        self.num_training_steps = 0
        self.num_warmup_steps = 0
        self.num_grad_accum_batches = 0
        self.validation_every_steps = 0
        self.save_every_steps = 0

        # state and progress
        self.next_save_step = 0
        self.num_trained_steps = 0
        self.next_val_step = 0
        self.next_accum_step = 0
        self.state_path = ""
        self.distr_device = ""
        self.saved_state = False
        self.val_batch_order_ids = []
        
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

    def _load_model(self, model_path: str = None):
        self._set_postfix("Loading model...")

        precision_dict = {
            "fp16": torch.float16,
            "4bit": torch.bfloat16,
            "8bit": torch.bfloat16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16
        }
        
        train_precision = precision_dict.get(self.training_precision_name, torch.float16)

        path = model_path if model_path is not None else self.model_path

        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map="balanced_low_0" if self.multi_gpu else self.device,
            torch_dtype=train_precision,
            load_in_4bit=self.training_precision_name == "4bit",
            load_in_8bit=self.training_precision_name == "8bit",
            attn_implementation="flash_attention_2"
        )
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
            betas=(0.7, 0.95),
            optimizer_name=self.optimizer_name,
            weight_decay=2e-4
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

        state = {
            'model': self.model.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict()
        }
        torch.save(state, os.path.join(self.paths.student_states, "training_state.pt"))

        self._release_postfix()

    def _save_model(self, step: int|str):
        self._set_postfix(f"Saving model at step {step}...")

        folder_name = f"{self.model_name}_step_{step}"
        self.model.save_pretrained(os.path.join(self.paths.student_trained, folder_name))
        self.tokenizer.save_pretrained(os.path.join(self.paths.student_trained, folder_name))

        self._release_postfix()

    def _load_state(self):
        self._set_postfix("Loading state...")

        state = torch.load(os.path.join(self.paths.student_states, "training_state.pt"))

        self._load_model()
        self.model.load_state_dict(state['model'], assign=True)

        self._load_optimizer()
        self.optimizer.load_state_dict(state['optimizer'])

        self.lr_scheduler = set_lr_scheduler(self.optimizer, self.lr_scheduler_name, self.num_warmup_steps, self.num_training_steps, self.dataset_len, self.decay_start, self.lr, self.final_lr) 
        self.lr_scheduler.load_state_dict(state['scheduler'])

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
            os.remove(os.path.join(self.paths.student_states, "training_state.pt"))
            self.saved_state = False
        else:
            self._load_model()
            self._load_optimizer()
            self.lr_scheduler = set_lr_scheduler(self.optimizer, self.lr_scheduler_name, self.num_warmup_steps, self.num_training_steps, self.dataset_len, self.decay_start, self.lr, self.final_lr) 

        if self.num_trained_steps == 0:
            self._validate(validation_data_manager)


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
                    
                self._run_training_cycle(data_list.pop(0), data_manager, validation_data_manager)
        else:
            self._reorder_dataset()
            dataset_chunk = self.dataset if full_collect else [convo for convo in self.dataset if convo.origin_convo_id in trainable_ids]
            batched_chunk_convos, id_batches = self._construct_batches(dataset_chunk)
            data_manager.enqueue_get_batches(id_batches)
            self._run_training_cycle(batched_chunk_convos, data_manager, validation_data_manager)


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
    def _run_training_cycle(self, batched_chunk_convos, data_manager: H5DataManager, validation_data_manager: H5DataManager):
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

            batch_logits = self.model(batch_tokenized_tensor).logits[:, :, :self.crop_to_size].float()

            if not self.distr_device:
                self.distr_device = batch_logits.device

            for i, convo in enumerate(batch_convos):
                indices = torch.from_numpy(batch_indices[i]).to(device, non_blocking=True)
                content_indices = self._get_content_indices_tensor(convo.content_ranges)

                convo_content_logits = torch.index_select(batch_logits[i], 0, content_indices) / self.temperature
                convo_teacher_logits = teacher_batch_raw[i]
                convo_content_tokens = torch.index_select(batch_tokenized_tensor[i], 0, content_indices)

                combined_loss, cross_entropy_loss, custom_loss, kl_div, variance, certainty_loss = calculate_divergence(convo_content_logits, convo_teacher_logits[:batch_padding[i]], indices, convo_content_tokens, custom=self.custom_reduction)

                batch_combined_loss += combined_loss / num_steps * self.num_grad_accum_batches
                batch_cross_entropy_loss += cross_entropy_loss / num_steps * self.num_grad_accum_batches
                batch_custom_loss += custom_loss / num_steps * self.num_grad_accum_batches
                batch_kl_div += kl_div / num_steps * self.num_grad_accum_batches
                batch_variance += variance / num_steps * self.num_grad_accum_batches
                batch_certainty_loss += certainty_loss / num_steps * self.num_grad_accum_batches

                self.lr_scheduler.step()
            
            multiplier = 1 / self.num_grad_accum_batches

            self.logger.log({"Loss/combined (custom + variance + certainty)": batch_combined_loss * multiplier}, step=self.num_trained_steps)
            self.logger.log({"Loss/cross entropy": batch_cross_entropy_loss * multiplier}, step=self.num_trained_steps)
            self.logger.log({"Loss/custom loss": batch_custom_loss * multiplier}, step=self.num_trained_steps)
            self.logger.log({"Loss/KL divergence": batch_kl_div * multiplier}, step=self.num_trained_steps)
            self.logger.log({"Loss/variance": batch_variance * multiplier}, step=self.num_trained_steps)
            self.logger.log({"Loss/certainty loss": batch_certainty_loss * multiplier}, step=self.num_trained_steps)

            self.logger.log({"Learning rate": self.lr_scheduler.get_last_lr()[0]}, step=self.num_trained_steps)
            self.progress_bar.update(num_steps)

            batch_combined_loss.backward()

            self.num_trained_steps += num_steps

            if self.num_trained_steps >= self.next_accum_step:
                self.optimizer.step()
                self.optimizer.zero_grad()
                self.next_accum_step += self.num_grad_accum_batches * self.batch_size

            if (self.num_trained_steps >= self.next_save_step) and not (self.num_trained_steps >= self.num_training_steps):
                self._save_model(self.num_trained_steps)
                self.next_save_step += self.save_every_steps

            if self.num_trained_steps >= self.next_val_step:
                self._validate(validation_data_manager)

    def _validate(self, validation_data_manager: H5DataManager):
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