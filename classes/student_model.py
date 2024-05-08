from utils.finetuning_utils import set_optimizer, set_lr_scheduler, calculate_divergence, launch_tensorboard
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.tensorboard.writer import SummaryWriter
from classes.data_classes import ConvoTokenized
from classes.data_manager import H5DataManager
from multiprocessing import shared_memory
from classes.base_model import BaseModel
from classes.paths import Paths
from typing import Optional
from tqdm import tqdm
import numpy as np
import torch
import math
import time
import os


class StudentModel(BaseModel):
    def __init__(self, model_path: str, paths: Paths):
        super().__init__(model_path)
        self.model: Optional[AutoModelForCausalLM] = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.data_order = ""
        self.optimizer_name = ""
        self.optimizer = None
        self.lr_scheduler_name = ""
        self.lr_scheduler = None
        self.lr = 0
        self.grad_accum_steps = 0
        self.num_training_steps = 0
        self.validation_per_steps = 0
        self.val_batch_order_ids = []
        self.save_interval = 0
        self.training_precision_name = ""
        self.decay_start = 0
        self.final_lr = 5e-7
        self.paths: Paths = paths
        self.num_epochs = 0
        self.num_warmup_steps = 0
        self.multi_gpu = False
        self.num_trained_steps = 0
        self.next_val_step = 0
        self.next_accum_step = 0
        self.custom_reduction = False
        self.logger = None
        self.tensorboard = None
        self.saved_state = False
        self.save_every_steps = 0
        self.next_save_step = 0
        self.state_path = ""
        self.distr_device = ""

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
            device_map="auto" if self.multi_gpu else self.device,
            torch_dtype=train_precision,
            load_in_4bit=self.training_precision_name == "4bit",
            load_in_8bit=self.training_precision_name == "8bit",
            attn_implementation="flash_attention_2"
        )
        self.model.train()

        self._release_postfix()

    def _unload_model(self):
        self.model = None

    def _load_optimizer(self):
        self._set_postfix("Loading optimizer...")

        self.optimizer = set_optimizer(
            self.model.parameters(),
            lr=self.lr,
            betas=(0.5, 0.9),
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
            self.logger = SummaryWriter(log_dir=os.path.join(self.paths.logging, comment))
            self.tensorboard = launch_tensorboard(self.paths.logging)

        if self.grad_accum_steps < self.batch_size:
            print("Warning: Grad accum steps < batch size. Setting accumulation steps = batch size.")
            self.grad_accum_steps = self.batch_size

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
            self._save_state()
            self.saved_state = True

        self.progress_bar.close()
        self._unload_model()
        self._unload_optimizer()
        torch.cuda.empty_cache()
        self.lr_scheduler = None

    def close(self):
        if self.logger is not None:
            self.logger.close()
        if self.tensorboard is not None:
            self.tensorboard.terminate()

    # main training loop
    def _run_training_cycle(self, batched_chunk_convos, data_manager: H5DataManager, validation_data_manager: H5DataManager):
        for batch_convos in batched_chunk_convos:
            state_updated = False
            num_steps = len(batch_convos)
            device = self.distr_device if self.distr_device else self.device

            sdh_mem_name, batch_shape, batch_dtype, batch_indices, batch_padding = data_manager.read_next_batch()
            shd_mem = shared_memory.SharedMemory(name=sdh_mem_name)
            teacher_batch_raw = torch.from_numpy(np.ndarray(batch_shape, dtype=batch_dtype, buffer=shd_mem.buf)).to(device, non_blocking=True)

            max_non_padded_len = max(convo.length for convo in batch_convos)
            batch_tokenized = np.array([convo.tokenized[:max_non_padded_len] for convo in batch_convos])
            batch_tokenized_tensor = torch.from_numpy(batch_tokenized).to(self.device, non_blocking=True)

            batch_kl_div = torch.tensor(0.0).to(device, non_blocking=True)

            batch_logits = self.model(batch_tokenized_tensor).logits[:, :, :self.crop_to_size].float()

            if not self.distr_device:
                self.distr_device = batch_logits.device

            for i, convo in enumerate(batch_convos):
                indices = torch.from_numpy(batch_indices[i]).to(device, non_blocking=True)
                content_indices = self._get_content_indices_tensor(convo.content_ranges)

                convo_content_logits = torch.index_select(batch_logits[i], 0, content_indices) / self.temperature
                convo_teacher_logits = teacher_batch_raw[i]
                kl_div = calculate_divergence(convo_content_logits, convo_teacher_logits[:batch_padding[i]], indices, custom=self.custom_reduction)
                batch_kl_div += kl_div

                self.lr_scheduler.step()
            
            batch_kl_div /= num_steps
            self.logger.add_scalar("LR", self.lr_scheduler.get_last_lr()[0], self.num_trained_steps)
            self.logger.add_scalar("Loss/train", batch_kl_div, self.num_trained_steps)
            self.progress_bar.update(num_steps)
            batch_kl_div.backward()

            self.num_trained_steps += num_steps

            if self.num_trained_steps >= self.next_accum_step:
                self.optimizer.step()
                self.optimizer.zero_grad()
                state_updated = True
                self.next_accum_step += self.grad_accum_steps

            if self.num_trained_steps >= self.next_save_step:
                self._save_model(self.num_trained_steps)
                self.next_save_step += self.save_every_steps

            if self.num_trained_steps >= self.next_val_step:
                self._validate(validation_data_manager)

        if not state_updated:
            self.optimizer.step()
            self.optimizer.zero_grad()

    def _validate(self, validation_data_manager: H5DataManager):
        self.model.eval()
        device = self.distr_device if self.distr_device else self.device
        pbar = tqdm(total=self.validation_dataset_len, desc="Validating", leave=False, smoothing=0.06)

        with torch.no_grad():
            total_val_kl_div = torch.tensor(0.0).to(device, non_blocking=True)
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

                    val_kl_div = calculate_divergence(val_convo_content_logits, val_convo_teacher_logits[:batch_padding[i]], val_indices, custom=self.custom_reduction)
                    total_val_kl_div += val_kl_div

                pbar.update(len(val_convo_batch))

        self.logger.add_scalar("Loss/val", total_val_kl_div, self.num_trained_steps)
        pbar.close()
        self.model.train()
        self.next_val_step += self.validation_per_steps