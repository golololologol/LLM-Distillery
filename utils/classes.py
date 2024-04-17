from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
from transformers import AutoModelForCausalLM, AutoTokenizer
from utils.vocab_utils import get_vocab_family, get_special_tokens
#from utils.finetuning_utils import set_optimizer
from utils.convert_to_safetensor import convert_model
from multiprocessing import shared_memory
from typing import Optional
from numpy import ndarray
from tqdm import tqdm
import torch.nn.functional as F
import numpy as np
import multiprocessing
import shutil
import torch
import time
import math
import json
import gc
import os


def is_model_safetensors(model_path: str):
    if os.path.isdir(model_path):
        for file in os.listdir(model_path):
            if file.endswith('.safetensors'):
                return True
    return False

def load_prompt_format(model_path: str) -> None | dict:
    prompt_format_path = os.path.join(model_path, "prompt_format.json")
    if not os.path.exists(prompt_format_path):
        return None
    
    with open(prompt_format_path, 'r', encoding='utf-8') as file:
        return json.load(file)
    
def save_prompt_format(prompt_format: dict, save_folder: str):
    prompt_format_path = os.path.join(save_folder, "prompt_format.json")
    with open(prompt_format_path, 'w', encoding='utf-8') as file:
        json.dump(prompt_format, file, ensure_ascii=False, indent=4)

def load_config(model_path):
    config_path = os.path.join(model_path, "pipeline_config.json")
    if not os.path.exists(config_path):
        return None
    
    with open(config_path, 'r', encoding='utf-8') as f:
        return json.load(f)
    
def save_config(config, model_path):
    config_path = os.path.join(model_path, "pipeline_config.json")
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(config, f, ensure_ascii=False, indent=4)

def input_prompt_format():
    prompt_format = {
        'SYS_START': "### System:\n",
        'USER_START': "### User:\n",
        'ASSISTANT_START': "### Assistant:\n",
        'SYS_END': "\n",
        'USER_END': "\n",
        'ASSISTANT_END': "\n"
    }

    keys = list(prompt_format.keys())
    i = 0
    print("Enter the prompt format, use '<' to go back a step.")
    while i < len(keys):
        key = keys[i]
        default_value = prompt_format[key].encode('unicode_escape').decode()
        value = input(f"{key} (default: {default_value}): ")
        
        if value == "<":
            i = max(0, i - 1)
        elif value == "":
            i += 1
        else:
            prompt_format[key] = value
            i += 1
            
    return prompt_format

def input_config():
    config = {
        'batch_size': 1,
        'add_bos': True,
        'seq_chunk_len': 256
    }
    print("Enter the model config, use '<' to go back a step.")
    keys = list(config.keys())
    i = 0
    while i < len(keys):
        key = keys[i]
        default_value = config[key]
        value_input = input(f"{key} (default: {default_value}): ")
        
        if value_input == "<":
            i = max(0, i - 1)
            continue
        elif value_input == "":
            i += 1
            continue
        
        # Type conversion magic here
        try:
            if isinstance(default_value, bool):
                config[key] = value_input.lower() in ['true', '1', 't', 'y', 'yes']
            elif isinstance(default_value, int):
                config[key] = int(value_input)
            else:
                config[key] = value_input
        except ValueError:
            print(f"You messed up. Enter a valid value for {key}.")
            continue  # Don't advance if the user's input was trash

        i += 1

    return config

class Paths:
    '''
    cache
    ├─ tensorboard_logs
    ├─ dataset
    │  └─ validation
    └─ student
       ├─ states
       ├─ gguf
       └─ trained
    '''

    def __init__(self, cache, clean_start: bool = False):
        self.cache: str = cache
        self.logging: str = os.path.join(cache, "tensorboard_logs")
        self.dataset: str = os.path.join(cache, "dataset")
        self.dataset_validation = os.path.join(self.dataset, "validation")
        self.student_root: str = os.path.join(cache, "student")
        self.student_states: str = os.path.join(self.student_root, "states")
        self.student_gguf: str = os.path.join(self.student_root, "gguf")
        self.student_trained: str = os.path.join(self.student_root, "trained")
        self.initialize_folders(clean_start)
    
    def initialize_folders(self, clean_start):
        if clean_start:
            self.empty_all()
        os.makedirs(self.cache, exist_ok=True)
        os.makedirs(self.logging, exist_ok=True)
        os.makedirs(self.dataset, exist_ok=True)
        os.makedirs(self.dataset_validation, exist_ok=True)
        os.makedirs(self.student_root, exist_ok=True)
        os.makedirs(self.student_states, exist_ok=True)
        os.makedirs(self.student_gguf, exist_ok=True)
        os.makedirs(self.student_trained, exist_ok=True)

    def create_folder(self, existing_folder, new_folder: str):
        os.makedirs(os.path.join(existing_folder, new_folder), exist_ok=True)
        setattr(self, new_folder, os.path.join(existing_folder, new_folder))

    def empty_folder(self, folder: str):
        shutil.rmtree(folder)

    def empty_dataset(self):
        self.empty_folder(self.dataset)
    
    def empty_student_folders(self):
        self.empty_folder(self.student_states)
        self.empty_folder(self.student_gguf)
        self.empty_folder(self.student_trained)

    def empty_logs(self):
        self.empty_folder(self.logging)

    def empty_all(self):
        self.empty_folder(self.cache)

class Distribution:
    def __init__(self, origin_convo_id: int, length: int = 0, cropped_end: bool = False, content_ranges: list[tuple[int, int]] = [], tokenized: ndarray = None):
        self.distribution: ndarray|torch.Tensor = None
        self.length: int = length
        self.tokenized: ndarray = tokenized
        self.origin_convo_id: int = origin_convo_id
        self.ppl: float = -1
        self.cropped_end: bool = cropped_end
        self.content_ranges: list[tuple[int, int]] = content_ranges
        self.shd_mem_name: str = ""
        self.distr_shape = None
        self.distr_dtype = None

class ConvoTokenized:
    def __init__(self, tokenized: ndarray, content_ranges, padding, cropped_end, convo_id):
        self.tokenized: ndarray = tokenized
        self.content_ranges: list[tuple[int, int]] = content_ranges
        self.padding: int = padding
        self.cropped_end: bool = cropped_end
        self.length: int = len(tokenized) - padding
        self.len_content: int = sum([end - start for start, end in content_ranges])
        self.origin_convo_id: int = convo_id

class BaseModel:
    def __init__(self, model_path: str):
        self.model_path: str = model_path
        self.model_name: str = ""
        self.device: str = "cuda:0"
        self.prompt_format: dict = {}

        self.batch_size: int = 0
        self.add_bos: bool = False
        self.context_len: int = 0
        self.seq_chunk_len: int = 0

        self.progress_bar: Optional[tqdm] = None

        self.dataset: list[ConvoTokenized] = []
        self.dataset_len: int = 0
        self.dataset_sorted: bool = False

        self.validation_dataset: list[ConvoTokenized] = []
        self.validation_dataset_len: int = 0
        self.validation_dataset_sorted: bool = False

        self.vocab_family: str = ""
        self.special_tokens: dict = {}
        self.temperature: float = 1.0
        self.crop_to_size: int = 0
        self._prepare()

    def _prepare(self):
        self.model_name = os.path.basename(self.model_path)

        if not os.path.exists(self.model_path):
            self.model_path = f"{self.model_path}_safetensors"

            if not os.path.exists(self.model_path):
                raise FileNotFoundError(f"Model path for {self.model_name} does not exist.")

        if not is_model_safetensors(self.model_path):
            self.model_path = convert_model(self.model_path)

        pf = load_prompt_format(self.model_path)
        if pf is None:
            print(f"{self.model_name} has no prompt format")
            pf = input_prompt_format()
            save_prompt_format(pf, self.model_path)

        config = load_config(self.model_path)
        if config is None:
            print(f"{self.model_name} has no config")
            config = input_config()
            save_config(config, self.model_path)

        self.prompt_format = pf
        self.batch_size = config.get('batch_size', 1)
        self.add_bos = config.get('add_bos', True)
        self.seq_chunk_len = config.get('seq_chunk_len', 256)
        self.vocab_family = get_vocab_family(model_path=self.model_path)
        self.special_tokens = get_special_tokens(self.vocab_family)

    def _get_content_indices_tensor(self, content_ranges) -> torch.Tensor:
        content_indices = []
        for start, end in content_ranges:
            content_indices.append(torch.arange(start, end, device=self.device))
        return torch.cat(content_indices)

class StudentModel(BaseModel):
    def __init__(self, model_path: str, paths: Paths):
        super().__init__(model_path)
        self.model: Optional[AutoModelForCausalLM] = None
        self.data_order = ""
        self.optimizer_name = ""
        self.optimizer = None
        self.validation_data_manager = None
        self.lr_scheduler_name = ""
        self.lr_scheduler = None
        self.lr = 0
        self.grad_accum_steps = 0
        self.num_training_steps = 0
        self.validation_per_steps = 0
        self.save_interval = 0
        self.training_precision_name = ""
        self.decay_start = 0
        self.paths: Paths = paths
        self.num_epochs = 0
        self.num_warmup_steps = 0
        self.multi_gpu = False

    def load_model(self):

        print(f"Loading {self.model_name}...")

        precision_dict = {
            "fp16": torch.float16,
            "4bit": torch.bfloat16,
            "fp32": torch.float32,
            "bf16": torch.bfloat16
        }
        
        train_precision = precision_dict.get(self.training_precision_name, torch.float16)

        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_path,
            device_map="auto" if self.multi_gpu else self.device,
            torch_dtype=train_precision,
            load_in_4bit=self.training_precision_name == "4bit",
            attn_implementation="flash_attention_2"
        )
        self.model.train()

    """def load_optimizer(self):
        self.optimizer = set_optimizer(
            self.model.parameters(),
            lr=self.lr,
            grad_accum_steps=self.grad_accum_steps,
            betas=(0.5, 0.9),
            optimizer_name=self.optimizer_name,
            weight_decay=2e-5
        )"""
    
    def sort_dataset_by_len(self):
        self.validation_dataset.sort(key=lambda convo: convo.length, reverse=True)
        self.validation_dataset_sorted = True
        validation_loading_order = [convo.origin_convo_id for convo in self.validation_dataset]

        self.dataset.sort(key=lambda convo: convo.length, reverse=True)
        self.dataset_sorted = True
        loading_order = [convo.origin_convo_id for convo in self.dataset]
        return loading_order, validation_loading_order
    
    def _order_dataset(self):
        if self.dataset_sorted:
            return
        if self.data_order == "shuffle":
            np.random.shuffle(self.dataset)
        elif self.data_order == "sorted":
            self.dataset.sort(key=lambda convo: convo.length, reverse=True)
            self.dataset_sorted = True
        elif self.data_order == "native":
            self.dataset_sorted = True
            pass

    def save_state(self):
        self.model.save_pretrained(self.paths.student_states)
        self.model.config.save_pretrained(self.paths.student_states)
        torch.save(self.optimizer.state_dict(), os.path.join(self.paths.student_states, "optimizer.pt"))
        torch.save(self.lr_scheduler.state_dict(), os.path.join(self.paths.student_states, "scheduler.pt"))

    def load_state(self):
        self.model = AutoModelForCausalLM.from_pretrained(self.paths.student_states)
        self.model.train()
        self.optimizer.load_state_dict(torch.load(os.path.join(self.paths.student_states, "optimizer.pt")))
        self.lr_scheduler.load_state_dict(torch.load(os.path.join(self.paths.student_states, "scheduler.pt")))

    def _construct_batches(self, dataset_chunk: list[ConvoTokenized]) -> list[list[ConvoTokenized]]:
        num_batches = math.ceil(len(dataset_chunk) / self.batch_size)
        convo_batches = []
        id_batches = []

        for i in range(num_batches):
            batch = dataset_chunk[i*self.batch_size:(i+1)*self.batch_size]
            convo_batches.append(batch)
            id_batches.append([convo.origin_convo_id for convo in batch])

        return convo_batches, id_batches

    def train_chunk(self, data_manager, validation_data_manager, full_collect):
        trainable_ids = data_manager.get_convo_ids()
        num_to_process = len(trainable_ids)

        self._order_dataset()
        dataset_chunk = self.dataset if full_collect else [convo for convo in self.dataset if convo.origin_convo_id in trainable_ids]
        convo_batches, id_batches = self._construct_batches(dataset_chunk)
        data_manager.enqueue_get_batches(id_batches)

        self.validation_data_manager = validation_data_manager


        






def _batch_creator_worker(inference_queue, batch_size, dataset_chunk: list[ConvoTokenized]):
    def _prepare_batch_for_inference(convo_id, batch_size, dataset_chunk: list[ConvoTokenized]) -> tuple[ndarray, list[Distribution], int]:
        batch_tokenized: list[ndarray] = []
        batch_distributions: list[Distribution] = []

        batch_convos = dataset_chunk[convo_id:convo_id+batch_size]

        for convo in batch_convos:
            batch_distributions.append(Distribution(convo.origin_convo_id, convo.length, convo.cropped_end, convo.content_ranges, convo.tokenized))
            batch_tokenized.append(convo.tokenized)

        # Crop to the convos to one that has the least non-padding tokens
        max_non_padded_len = max(convo.length for convo in batch_convos)
        batch_tokenized = [convo_tokenized[:max_non_padded_len] for convo_tokenized in batch_tokenized]

        batch_tokenized_np = np.array(batch_tokenized)
        batch_size = len(batch_distributions)

        return batch_tokenized_np, batch_distributions, batch_size
        
    convo_id = 0
    num_batches = math.ceil(len(dataset_chunk) / batch_size)

    for i in range(num_batches):
        batch_tokenized, batch_distributions, batch_size = _prepare_batch_for_inference(convo_id, batch_size, dataset_chunk)
        inference_queue.put((batch_tokenized, batch_distributions))
        convo_id += batch_size

    inference_queue.put((None, None))

def _inference_worker(inference_queue, result_queue, made_distributions, done_chunk, model_path, model_name, reserve_vram,
                        device, temperature, crop_to_size, pbar_queue, context_len, batch_size, max_queue_size, seq_chunk_len):
        
    def _load_model(reserve_vram_gb: list[float] = []):
        pbar_queue.put(("str", f"Loading {model_name}..."))

        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise Exception("No CUDA-capable GPUs found.")

        reserve_vram_kb = [int(128 * 1000**2)]*num_gpus # default 128MB/GPU

        for i, reserve_gb in enumerate(reserve_vram_gb):
            if reserve_gb > 0 and i < num_gpus:
                reserve_vram_kb[i] = int(reserve_gb * 1000**3)

        config = ExLlamaV2Config()
        config.model_dir = model_path
        config.prepare()
        config.max_seq_len = context_len
        config.max_batch_size = batch_size
        config.max_input_len = seq_chunk_len
        config.max_attention_size = context_len ** 2 

        model = ExLlamaV2(config)

        cache = ExLlamaV2Cache(model, batch_size, context_len, lazy=True)

        model.load_autosplit(cache, reserve_vram=reserve_vram_kb)
        return model, cache

    def _unload_model(model: ExLlamaV2, cache: ExLlamaV2Cache):
        if model is None:
            print(f"{model_name} is already unloaded, genius.")
            return

        model.unload()

        del model
        del cache
        torch.cuda.empty_cache()

    def _inference(model: ExLlamaV2, cache: ExLlamaV2Cache, batch_tokenized: torch.Tensor) -> ndarray:
        assert model is not None, f"{model_name}: Model has not been loaded yet."

        with torch.no_grad():
            cache.current_seq_len = 0
            
            batch_logp: torch.Tensor = F.log_softmax(model.forward(batch_tokenized, cache=cache).contiguous()[:, :, :crop_to_size].float()/temperature, dim=-1)

            batch_logp_data: ndarray = batch_logp.to('cpu').numpy()
            shd_mem = shared_memory.SharedMemory(create=True, size=batch_logp_data.nbytes)
            shared_batch_logp = ndarray(batch_logp_data.shape, dtype=batch_logp_data.dtype, buffer=shd_mem.buf)
            np.copyto(shared_batch_logp, batch_logp_data)
            return shd_mem.name, batch_logp_data.shape, batch_logp_data.dtype, shd_mem
        
    def _enqueue_batch_tensor():
        batch_tokenized_np, batch_distributions = inference_queue.get()

        if batch_tokenized_np is None:
            tokenized_batches.append((None, None))
            return

        batch_tensor = torch.tensor(batch_tokenized_np, dtype=torch.long).to(device, non_blocking=True)
        tokenized_batches.append((batch_tensor, batch_distributions))
        
    model, cache = _load_model(reserve_vram)

    pbar_queue.put(("str", f"Generating..."))

    tokenized_batches = []
    tokenized_batches_max = math.ceil(100 / batch_size)
    shared_list = []

    while not inference_queue.empty() and len(tokenized_batches) < tokenized_batches_max:
        _enqueue_batch_tensor()

    while True:
        batch_tensor, batch_distributions = tokenized_batches.pop(0)

        if batch_tensor is None:
            _unload_model(model, cache)
            result_queue.put((None, None, None, None))
            made_distributions.set()
            done_chunk.wait()
            break

        shd_mem_name, batch_logp_shape, batch_logp_dtype, shd_mem = _inference(model, cache, batch_tensor)
        shared_list.append(shd_mem)
        shared_list = shared_list[-(4 + max_queue_size):]
        result_queue.put((shd_mem_name, batch_logp_shape, batch_logp_dtype, batch_distributions))
        made_distributions.set()

        if not inference_queue.empty():
            _enqueue_batch_tensor()

def _result_processor_worker(result_queue, made_distributions, done_chunk_writes, disk_queue, encourage_eos, special_tokens, pbar_queue: tqdm, max_queue_size: int, batch_size: int):
    def _get_content_indices_np(content_ranges) -> ndarray:
        content_indices = []
        for start, end in content_ranges:
            content_indices.append(np.arange(start, end))
        return np.concatenate(content_indices)
        
    def _get_batch_content_logprobs(batch_distributions: list[Distribution]) -> list[Distribution]:
        batch_content_distributions = []
        batch_distributions_len = len(batch_distributions)

        for distribution in batch_distributions:
            content_indices = _get_content_indices_np(distribution.content_ranges)
            distribution.distribution = distribution.distribution[content_indices]
            content_tokens = distribution.tokenized[content_indices][1:]
            gathered_log_probs = distribution.distribution[np.arange(len(content_indices) - 1), content_tokens]
            distribution.ppl = min(np.exp(-np.mean(gathered_log_probs)), 100)

            shd_mem = shared_memory.SharedMemory(create=True, size=distribution.distribution.nbytes)
            distribution.shd_mem_name = shd_mem.name
            distribution.distr_shape = distribution.distribution.shape
            distribution.distr_dtype = distribution.distribution.dtype
            shd_distr = np.ndarray(distribution.distr_shape, dtype=distribution.distr_dtype, buffer=shd_mem.buf)
            np.copyto(shd_distr, distribution.distribution)
            shared_list.append(shd_mem)
            del distribution.distribution

            batch_content_distributions.append(distribution)

        return batch_content_distributions, batch_distributions_len

    def _process_inference_results(batch_logprobs_np: ndarray, batch_distributions: list[Distribution]) -> int:
        for i, distribution in enumerate(batch_distributions):
            convo_logprobs = batch_logprobs_np[i]
            if encourage_eos:
                eos_id = special_tokens['eos_id']
                content_end_ids = [end-1 for start, end in distribution.content_ranges]

                if distribution.cropped_end:
                    content_end_ids = content_end_ids[:-1]

                for end in content_end_ids:
                    convo_logprobs[end, eos_id] = np.log((np.max(np.exp(convo_logprobs[end])) * 1.1))
                    convo_logprobs[end] = np.log((np.exp(convo_logprobs[end]) / np.sum(np.exp(convo_logprobs[end]))))

            distribution.distribution = convo_logprobs[:distribution.length]
        batch_content_distributions, batch_distributions_len = _get_batch_content_logprobs(batch_distributions)

        disk_queue.put(batch_content_distributions)

        return batch_distributions_len
    
    exit_flag = False
    shared_list = []

    while True:
        made_distributions.wait()
        while not result_queue.empty():
            shd_mem_name, batch_logp_shape, batch_logp_dtype, batch_distributions = result_queue.get()

            if batch_distributions is None:
                exit_flag = True
                done_chunk_writes.set()
                break

            logp_shd_mem = shared_memory.SharedMemory(name=shd_mem_name)
            batch_logprobs_np = np.ndarray(batch_logp_shape, dtype=batch_logp_dtype, buffer=logp_shd_mem.buf)

            batch_distributions_len = _process_inference_results(batch_logprobs_np, batch_distributions)
            shared_list = shared_list[-((1 + max_queue_size)*batch_size):]
            logp_shd_mem.close()
            logp_shd_mem.unlink()

            pbar_queue.put(("increment", batch_distributions_len))
        
        made_distributions.clear()

        if exit_flag:
            # block until terminated
            made_distributions.wait()

class TeacherModel(BaseModel):
    def __init__(self, model_path: str, max_queue_size=3):
        super().__init__(model_path)
        self.distr_device: str = ""
        self.encourage_eos: bool = False
        self.stop_id: int = 0
        self.reserve_vram = []
        self.max_queue_size = max_queue_size

    def process_chunk(self, reserve_vram_gb: list[float] = [], next_stop_id: int = 1, data_manager = None, full_collect: bool = False, validation: bool = False):
        if full_collect:
            num_to_process = self.validation_dataset_len if validation else self.dataset_len
        else:
            num_to_process = next_stop_id - self.stop_id

        dataset_chunk = (self.validation_dataset if validation else self.dataset)[self.stop_id:self.stop_id+num_to_process]

        with multiprocessing.Manager() as manager:
            done_chunk = manager.Event()
            made_distributions = manager.Event()
            inference_queue = manager.Queue(self.max_queue_size)
            result_queue = manager.Queue(self.max_queue_size)
            disk_queue = manager.Queue(self.max_queue_size)
            pbar_queue = manager.Queue(self.max_queue_size)

            self.progress_bar = tqdm(total=num_to_process, desc="Convos", smoothing=0.06, leave=False)
            self.data_manager = data_manager
            self.reserve_vram = reserve_vram_gb

            workers = self._start_workers(done_chunk, made_distributions, inference_queue, result_queue, disk_queue, pbar_queue, dataset_chunk)

            while not done_chunk.is_set():
                self._manage_queues(disk_queue, pbar_queue)

            self.data_manager.done_everything.wait()
            self._stop_workers(*workers)
            self.progress_bar.close()
            self.stop_id = next_stop_id

    def _manage_queues(self, disk_queue, pbar_queue):
        while not disk_queue.empty():
            batch_content_distributions = disk_queue.get()
            self.data_manager.write_batch(batch_content_distributions)

        while not pbar_queue.empty():
            action, value = pbar_queue.get()
            self._pbar_actions(action, value)

    def sort_dataset_by_len(self):
        self.validation_dataset.sort(key=lambda convo: convo.length, reverse=True)
        self.validation_dataset_sorted = True
        validation_sorting_map = {convo.origin_convo_id: i for i, convo in enumerate(self.validation_dataset)}

        self.dataset.sort(key=lambda convo: convo.length, reverse=True)
        self.dataset_sorted = True
        sorting_map = {convo.origin_convo_id: i for i, convo in enumerate(self.dataset)}
        return sorting_map, validation_sorting_map
    
    def sort_dataset_by_map(self, sorting_map: dict[int, int], validation: bool = False):
        if validation:
            self.validation_dataset.sort(key=lambda convo: sorting_map[convo.origin_convo_id])
            self.validation_dataset_sorted = True
        else:
            self.dataset.sort(key=lambda convo: sorting_map[convo.origin_convo_id])
            self.dataset_sorted = True

    def _pbar_actions(self, action: str, value: int):
        if action == "increment":
            self.progress_bar.update(value)
        elif action == "str":
            self.progress_bar.set_postfix_str(value)
    
    def _unload_full(self):
        self.progress_bar.set_postfix_str(f"Fully unloading {self.model_name}...")
        self._stop_workers()
        self.dataset = []
        self.dataset_len = 0
        self.validation_dataset = []
        self.convo_id = 0
        self.stop_id = 0
        self.dataset_sorted = False
        self.validation_dataset_sorted = False
        gc.collect()
    
    def _start_workers(self, done_chunk, made_distributions, inference_queue, result_queue, disk_queue, pbar_queue, dataset_chunk: list[ConvoTokenized]):
        self.progress_bar.set_postfix_str(f"Starting workers for {self.model_name}...")

        batch_creator = multiprocessing.Process(target=_batch_creator_worker, args=(inference_queue, self.batch_size, dataset_chunk))
        inference_worker = multiprocessing.Process(target=_inference_worker, args=(inference_queue, result_queue, made_distributions, done_chunk, self.model_path, self.model_name, self.reserve_vram,
                                                                                         self.device, self.temperature, self.crop_to_size, pbar_queue, self.context_len, self.batch_size, self.max_queue_size, self.seq_chunk_len))
        result_processor = multiprocessing.Process(target=_result_processor_worker, args=(result_queue, made_distributions, done_chunk, disk_queue, self.encourage_eos, self.special_tokens, pbar_queue, self.max_queue_size, self.batch_size))

        batch_creator.start()
        inference_worker.start()
        result_processor.start()

        return batch_creator, inference_worker, result_processor
    
    def _stop_workers(self, batch_creator: multiprocessing.Process, inference_worker: multiprocessing.Process, result_processor: multiprocessing.Process):
        self.progress_bar.set_postfix_str(f"Stopping workers for {self.model_name}...")
        batch_creator.join()
        inference_worker.join()
        result_processor.terminate()
    
    def write_dataset_to_file(self, folder: str):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        for convo in self.dataset[:4]:
            convo_dict = {
                "tokenized": convo.tokenized.tolist(),
                "decoded": [tokenizer.decode(convo.tokenized)],
                "content_ranges": convo.content_ranges,
                "content_decoded": [tokenizer.decode(convo.tokenized[start:end]) for start, end in convo.content_ranges],
                "padding": convo.padding,
                "cropped_end": convo.cropped_end,
                "origin_convo_id": convo.origin_convo_id
            }
            convo_path = os.path.join(folder, f"{convo.origin_convo_id}.json")
            with open(convo_path, 'w', encoding='utf-8') as file:
                json.dump(convo_dict, file, ensure_ascii=False, indent=4)