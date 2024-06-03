from utils.finetuning_utils import set_optimizer, set_lr_scheduler, calculate_divergence
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
from transformers import AutoModelForCausalLM, AutoTokenizer
from classes.data_classes import ConvoTokenized
from classes.data_manager import H5DataManager
from multiprocessing import shared_memory
from classes.base_model import BaseModel
from classes.losses import Losses
from classes.paths import Paths
from typing import Optional
from tqdm import tqdm
import torch.nn.functional as F
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

        self.model: Optional[AutoModelForCausalLM] = None
        self.adapter = None
        self.tokenizer: Optional[AutoTokenizer] = None
        self.add_bos = add_bos
        self.prompt_format = prompt_format
        self.batch_size = batch_size
        self.paths: Paths = paths
        self.losses: Losses = None

        self.optimizer_name = ""
        self.optimizer = None
        self.lr_scheduler_name = ""
        self.lr_scheduler = None
        self.lr = 0.0
        self.decay_start = 0.0 # wsd only
        self.final_lr = 5e-7 # wsd only
        self.alpha: float = 1

        self.num_epochs = 0
        self.total_training_steps = 0
        self.num_warmup_steps = 0
        self.grad_accum = 0
        self.num_grad_accum_batches = 0
        self.eff_batch_size = 0
        self.validation_every_steps = 0
        self.save_every_steps = 0

        # state and progress tracking
        self.next_save_step = 0
        self.num_trained_steps = 0
        self.next_val_step = 0
        self.next_accum_step = 0
        self.state_path = ""
        self.saved_state = False
        self.val_batch_order_ids = []
        
        # misc
        self.freeze_layers = []
        self.training_precision_name = ""
        self.logger = None
        self.data_order = ""
        self.save_final_state = False
        self.grad_checkpointing = False
        self.multi_gpu = False
        self.wandb_comment = ""
        self.device_map_name = ""
        self.device_map = {}
        self.max_memory = {}
        self.distr_device = ""
        self.input_device = ""
        self.num_gpu0_layers = 0
        

    def _set_postfix(self, postfix: str):
        self.progress_bar.set_postfix_str(postfix)

    def _release_postfix(self):
        self.progress_bar.set_postfix_str("")

    def _get_content_indices_tensor(self, content_ranges, device) -> torch.Tensor:
        content_indices = []
        for start, end in content_ranges:
            content_indices.append(np.arange(start, end))
        return torch.as_tensor(np.concatenate(content_indices), dtype=torch.long).to(device, non_blocking=True)

    def _load_model(self, model_path: str = None, load_empty: bool = False):
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

        if self.device_map:
            device_map = self.device_map
        else:
            device_map = self.device_map_name if self.multi_gpu else self.device

        if not self.device_map and self.device_map_name == "custom":
            model = AutoModelForCausalLM.from_pretrained(
                path,
                device_map="auto",
                torch_dtype=train_precision,
                load_in_4bit=self.training_precision_name == "4bit",
                load_in_8bit = self.training_precision_name == "8bit",
                max_memory = self.max_memory
                )

            layer_names = list(model.hf_device_map.keys())

            del model
            torch.cuda.empty_cache()

            num_layers = len(layer_names)
            num_gpus = torch.cuda.device_count()

            # Distribute layers across GPUs
            custom_device_map = {}
            remaining_layers = num_layers - self.num_gpu0_layers
            layers_per_gpu = math.ceil(remaining_layers / (num_gpus - 1))

            # Assign layers to GPU 0
            layer_index = 0
            for _ in range(self.num_gpu0_layers):
                custom_device_map[layer_names[layer_index]] = 0
                layer_index += 1

            # Assign remaining layers to other GPUs
            gpu_id = 1
            while layer_index < num_layers:
                for _ in range(layers_per_gpu):
                    if layer_index >= num_layers:
                        break
                    custom_device_map[layer_names[layer_index]] = gpu_id
                    layer_index += 1
                gpu_id += 1

            self.device_map = custom_device_map
            device_map = custom_device_map

        self.model = AutoModelForCausalLM.from_pretrained(
            path,
            device_map=device_map,
            torch_dtype=train_precision,
            load_in_4bit=self.training_precision_name == "4bit",
            load_in_8bit = self.training_precision_name == "8bit",
            attn_implementation="flash_attention_2",
            max_memory = self.max_memory
        )

        if not self.device_map:
            self.device_map = self.model.hf_device_map

        if not self.input_device:
            first_layer = list(self.device_map.keys())[0]
            self.input_device = f"cuda:{self.device_map[first_layer]}"

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
            weight_decay=2e-8
        )

        self._release_postfix()

    def _unload_optimizer(self):
        self.optimizer = None
    
    def _sort_val_dataset_by_len(self):
        self.validation_dataset.sort(key=lambda convo: convo.length, reverse=True)
        self.validation_dataset_sorted = True
    
    def reorder_dataset(self):
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

        if self.num_trained_steps < self.total_training_steps:
            self.paths.empty_student_states()

        torch.save(self.model.state_dict(), os.path.join(self.paths.student_states, "model_state.pt"))
        torch.save(self.optimizer.state_dict(), os.path.join(self.paths.student_states, "optimizer_state.pt"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(self.paths.student_states, "scheduler_state.pt"))

        self._release_postfix()

    def _save_model(self, step: int|str):
        self._set_postfix(f"Saving model at step {step}...")

        folder_name = f"{self.model_name}_step_{step}"

        self.model.save_pretrained(os.path.join(self.paths.student_trained, folder_name))

        self.tokenizer.save_pretrained(os.path.join(self.paths.student_trained, folder_name))

        self._release_postfix()

    def _load_state(self):
        self._set_postfix("Loading state...")

        self._load_model()
        model_state = torch.load(os.path.join(self.paths.student_states, "model_state.pt"))
        self.model.load_state_dict(model_state, assign=True)

        self._load_optimizer()
        optimizer_state = torch.load(os.path.join(self.paths.student_states, "optimizer_state.pt"))
        self.optimizer.load_state_dict(optimizer_state)

        self.lr_scheduler = set_lr_scheduler(self.optimizer, self.lr_scheduler_name, self.num_warmup_steps, self.num_grad_accum_batches, self.dataset_len, self.decay_start, self.lr, self.final_lr) 
        scheduler_state = torch.load(os.path.join(self.paths.student_states, "scheduler_state.pt"))
        self.lr_scheduler.load_state_dict(scheduler_state)

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
    
    def _prefetch_dataset_chunk(self, data_manager: H5DataManager, data_list: list, full_collect, trainable_ids) -> list:
            self.reorder_dataset()
            dataset_chunk = self.dataset if full_collect else [convo for convo in self.dataset if convo.origin_convo_id in trainable_ids]
            batched_chunk_convos, id_batches = self._construct_batches(dataset_chunk)
            data_list.append(batched_chunk_convos)
            data_manager.enqueue_get_batches(id_batches)
            return data_list

    def train_chunk(self, data_manager: H5DataManager, validation_data_manager: H5DataManager, full_collect):
        trainable_ids = data_manager.get_dataset_ids()
        self.distr_device = ""

        if not self.validation_dataset_sorted:
            self._sort_val_dataset_by_len()
            self.validation_dataset_batched, self.val_batch_order_ids = self._construct_batches(self.validation_dataset)
            validation_data_manager.read_only_mode(self.val_batch_order_ids)

        if self.logger is None:
            name = f"{self.wandb_comment} " if self.wandb_comment else ""
            model_name = self.model_name if len(self.model_name) <= 20 else f"{self.model_name[:20]}..."
            name += f"{model_name} lr({self.lr}) ({time.strftime('%d.%m.%Y / %H:%M:%S')})"
            self.logger = wandb.init(project="student_training", name=name, config=self.__dict__, group=self.model_name, reinit=True, dir=self.paths.cache)

        if self.tokenizer is None:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)

        self.progress_bar = tqdm(total=self.total_training_steps, initial=self.num_trained_steps, desc="Training", smoothing=0.06, leave=False)

        if self.saved_state:
            self._load_state()
            self.paths.empty_student_states()
            self.saved_state = False
        else:
            self._load_model()
            self._load_optimizer()
            self.lr_scheduler = set_lr_scheduler(self.optimizer, self.lr_scheduler_name, self.num_warmup_steps, self.num_grad_accum_batches, self.dataset_len, self.decay_start, self.lr, self.final_lr) 

        if self.losses is None:
            self.losses = Losses(self.logger)

        if self.num_trained_steps == 0:
            self._validate_distillation(validation_data_manager)
        
        # Main training loop
        if full_collect:
            data_manager_left_shuffles = self.num_epochs - 1
            data_list = self._prefetch_dataset_chunk(data_manager, [], full_collect, trainable_ids)
                
            for i in range(self.num_epochs):
                if data_manager_left_shuffles > 0:
                    data_list = self._prefetch_dataset_chunk(data_manager, data_list, full_collect, trainable_ids)
                    data_manager_left_shuffles -= 1

                updated_grads = self._run_distillation_cycle(data_list.pop(0), data_manager, validation_data_manager)
        else:
            data_list = self._prefetch_dataset_chunk(data_manager, [], full_collect, trainable_ids)
            updated_grads = self._run_distillation_cycle(data_list.pop(0), data_manager, validation_data_manager)

        if not updated_grads:
            self.optimizer.step()
            self.optimizer.zero_grad()
            self.losses.log(self.num_trained_steps)
            self.losses.empty()

        if not full_collect:
            self._save_state()
            self.saved_state = True

        if self.num_trained_steps >= self.total_training_steps:
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
            time.sleep(2)
            self.logger.finish()

    # main training loop
    def _run_distillation_cycle(self, batched_chunk_convos, data_manager: H5DataManager, validation_data_manager: H5DataManager) -> bool:
        for batch_convos in batched_chunk_convos:
            updated_grads = False
            batch_steps = len(batch_convos)
            logits_device = self.distr_device if self.distr_device else self.device

            sdh_mem_name, batch_shape, batch_dtype, batch_indices, batch_padding = data_manager.read_next_batch()
            shd_mem = shared_memory.SharedMemory(name=sdh_mem_name)
            teacher_batch_raw = torch.from_numpy(np.ndarray(batch_shape, dtype=batch_dtype, buffer=shd_mem.buf)).to(logits_device, non_blocking=True)

            max_non_padded_len = max(convo.length for convo in batch_convos)
            batch_tokenized = np.array([convo.tokenized[:max_non_padded_len] for convo in batch_convos])
            batch_tokenized_tensor = torch.from_numpy(batch_tokenized).to(self.input_device, non_blocking=True)

            batch_logits = self.model(batch_tokenized_tensor).logits[:, :, :self.crop_to_size].float()
            
            if not self.distr_device:
                self.distr_device = batch_logits.device
                logits_device = self.distr_device
                teacher_batch_raw = teacher_batch_raw.to(logits_device, non_blocking=True)
            
            for i, convo in enumerate(batch_convos):
                indices = torch.from_numpy(batch_indices[i]).to(logits_device, non_blocking=True)
                content_indices = self._get_content_indices_tensor(convo.content_ranges, logits_device)
                
                convo_content_logits = torch.index_select(batch_logits[i], 0, content_indices) / self.temperature
                convo_teacher_logits = teacher_batch_raw[i]
                convo_content_tokens = torch.index_select(batch_tokenized_tensor[i], 0, content_indices)

                loss_dict = calculate_divergence(convo_content_logits, convo_teacher_logits[:batch_padding[i]], indices, convo_content_tokens, self.alpha)

                self.losses.add_losses(loss_dict)
            
            self.losses.backward(divisor=self.eff_batch_size)
            
            self.progress_bar.update(batch_steps)

            self.num_trained_steps += batch_steps

            if self.num_trained_steps >= self.next_accum_step:
                self.optimizer.step()
                self.optimizer.zero_grad()

                self.logger.log({"Learning Rate": self.optimizer.param_groups[0]['lr']}, step=self.num_trained_steps)
                self.losses.log(self.num_trained_steps)
                self.losses.empty()

                self.lr_scheduler.step()

                updated_grads = True
                self.next_accum_step += self.eff_batch_size
                
            if (self.num_trained_steps >= self.next_save_step) and not (self.num_trained_steps >= self.total_training_steps):
                self._save_model(self.num_trained_steps)
                self.next_save_step += self.save_every_steps

            if self.num_trained_steps >= self.next_val_step:
                self._validate_distillation(validation_data_manager)

        return updated_grads

    def _validate_distillation(self, validation_data_manager: H5DataManager):
        self.model.eval()
        pbar = tqdm(total=self.validation_dataset_len, desc="Validating", leave=False, smoothing=0.06)
        losses = Losses(self.logger, validation=True)

        with torch.no_grad():
            for val_convo_batch in self.validation_dataset_batched:
                logits_device = self.distr_device if self.distr_device else self.device

                sdh_mem_name, batch_shape, batch_dtype, val_batch_indices, batch_padding = validation_data_manager.read_next_batch()
                shd_mem = shared_memory.SharedMemory(name=sdh_mem_name)
                teacher_batch_raw = torch.from_numpy(np.ndarray(batch_shape, dtype=batch_dtype, buffer=shd_mem.buf)).to(logits_device, non_blocking=True)

                val_max_non_padded_len = max(convo.length for convo in val_convo_batch)
                val_batch_tokenized = np.array([convo.tokenized[:val_max_non_padded_len] for convo in val_convo_batch])
                val_batch_tokenized_tensor = torch.from_numpy(val_batch_tokenized).to(self.input_device, non_blocking=True)

                val_batch_logits = self.model(val_batch_tokenized_tensor).logits[:, :, :self.crop_to_size].float()
                
                if not self.distr_device:
                    self.distr_device = val_batch_logits.device
                    logits_device = self.distr_device
                    teacher_batch_raw = teacher_batch_raw.to(logits_device, non_blocking=True)

                for i, val_convo in enumerate(val_convo_batch):
                    val_indices = torch.from_numpy(val_batch_indices[i]).to(logits_device, non_blocking=True)
                    val_content_indices = self._get_content_indices_tensor(val_convo.content_ranges, logits_device)

                    val_convo_content_logits = torch.index_select(val_batch_logits[i], 0, val_content_indices) / self.temperature
                    val_convo_teacher_logits = teacher_batch_raw[i][:batch_padding[i]]
                    val_convo_content_tokens = torch.index_select(val_batch_tokenized_tensor[i], 0, val_content_indices)
                    
                    loss_dict = calculate_divergence(val_convo_content_logits, val_convo_teacher_logits, val_indices, val_convo_content_tokens, self.alpha)
                    
                    losses.add_losses(loss_dict)
                    
                pbar.update(len(val_convo_batch))

        losses.log(self.num_trained_steps)
        losses.empty()

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


        