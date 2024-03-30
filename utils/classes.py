from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
from transformers import AutoModelForCausalLM, AutoTokenizer
from numpy import ndarray
from typing import Optional
from utils.vocab_utils import get_vocab_family, get_special_tokens
from utils.convert_to_safetensor import convert_model
from tqdm import tqdm
import multiprocessing
import pickle
import torch
import numpy as np
import shutil
import gc
import os
import json

def is_model_safetensors(model_path: str):
    if os.path.isdir(model_path):
        for file in os.listdir(model_path):
            if file.endswith('.safetensors'):
                return True
        return False
    
def load_prompt_format(model_path: str) -> None|dict:
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
        'context_chunk_size': 1024
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
        except ValueError as err:
            print(f"You messed up. Enter a valid value for {key}.")
            continue  # Don't advance if the user's input was trash

        i += 1

    return config

class Paths:
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
        self.context_chunk_size: int = 0

        self.data_manager = None
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
        self.convo_id: int = 0
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
        self.context_chunk_size = config.get('context_chunk_size', 1024)
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
        self.optimizer_name = ""
        self.optimizer = None
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
    
    def _inference(self, batch_tokenized: list[np.ndarray]) -> torch.Tensor:
        assert self.model is not None, f"{self.model_name}: Model has not been loaded yet."

        batch_tensor = torch.tensor(batch_tokenized, dtype=torch.long, device=self.device)
        batch_logits = self.model(batch_tensor).logits.float() / self.temperature
        return batch_logits
    
    def _get_batch_logprobs(self) -> list[Distribution]:
        with torch.no_grad():
            batch_distributions: list[Distribution] = []
            batch_tokenized = []
            for convo in self.validation_dataset:
                batch_distributions.append(Distribution(convo.origin_convo_id, convo.length))
                batch_tokenized.append(convo.tokenized)
            
            batch_size = len(batch_distributions)
            max_non_padded_len = max([convo.length for convo in self.validation_dataset])
            batch_tokenized = [convo_tokenized[:max_non_padded_len] for convo_tokenized in batch_tokenized]
            batch_logits = self._inference(batch_tokenized)
            batch_logprobs = torch.nn.functional.log_softmax(batch_logits, dim=-1)

            for i in range(0, batch_size):
                convo_logprobs = batch_logprobs[i]
                content_indices = self._get_content_indices_tensor(self.validation_dataset[i].content_ranges)
                convo_logprobs = convo_logprobs[content_indices]
                batch_distributions[i].distribution = convo_logprobs.cpu().numpy()
                batch_distributions[i].ppl = min(torch.exp(-(convo_logprobs.mean())), 100)
                
            return batch_distributions
        
    def _init_logging(self):
        self.paths.empty_logs()
        self.paths.empty_student_folders()

def _batch_creator_worker(made_tokens, inference_queue, batch_size, stop_id, dataset: list[ConvoTokenized]):
    def _prepare_batch_for_inference(convo_id, batch_size, stop_id, dataset: list[ConvoTokenized]) -> tuple[ndarray, list[Distribution], int]:
        batch_tokenized: list[np.ndarray] = []
        batch_distributions: list[Distribution] = []

        convos_to_inference = min(convo_id + batch_size, stop_id)
        batch_convos = dataset[convo_id:convos_to_inference]

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
    while convo_id < stop_id:
        batch_tokenized, batch_distributions, batch_size = _prepare_batch_for_inference(convo_id, batch_size, stop_id, dataset)
        inference_queue.put((batch_tokenized, batch_distributions))
        convo_id += batch_size
        made_tokens.set()
    inference_queue.put((None, None))
    made_tokens.set()

def _inference_worker(inference_queue, result_queue, made_tokens, made_distributions, model_path, model_name, reserve_vram,
                        device, temperature, crop_to_size, pbar_queue, context_len, context_chunk_size,batch_size):
        
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
        config.max_input_len = context_len
        config.max_attention_size = context_len ** 2 

        model = ExLlamaV2(config)
        cache = ExLlamaV2Cache(model, lazy=True)

        model.load_autosplit(cache, reserve_vram=reserve_vram_kb)
        return model

    def _unload_model(model: ExLlamaV2):
        if model is None:
            print(f"{model_name} is already unloaded, genius.")
            return

        model.unload()

        del model
        torch.cuda.empty_cache()

    def _inference(model: ExLlamaV2, batch_tokenized: torch.Tensor) -> ndarray:
        assert model is not None, f"{model_name}: Model has not been loaded yet."

        with torch.no_grad():
            batch_logprobs: torch.Tensor = torch.nn.functional.log_softmax(model.forward(batch_tokenized).float()/temperature, dim=-1)

            return batch_logprobs.cpu().numpy()[:, :, :crop_to_size]
        
    model = _load_model(reserve_vram)

    pbar_queue.put(("str", f"Generating..."))

    exit_flag = False

    while True:
        made_tokens.wait()
        while not inference_queue.empty():
            batch_tokenized_np, batch_distributions = inference_queue.get()

            if batch_tokenized_np is None:
                _unload_model(model)
                result_queue.put((None, None))
                made_distributions.set()
                exit_flag = True
                break
            
            batch_tensor = torch.tensor(batch_tokenized_np, dtype=torch.long, device=device)
            batch_logprobs_np = _inference(model, batch_tensor)
            result_queue.put((batch_logprobs_np, batch_distributions))
            made_distributions.set()

        made_tokens.clear()
        if exit_flag:
            break

def _result_processor_worker(result_queue, made_distributions, done_chunk_writes, disk_queue, encourage_eos, special_tokens, pbar_queue: tqdm):
    def _get_content_indices_np(content_ranges) -> ndarray:
        content_indices = []
        for start, end in content_ranges:
            content_indices.append(np.arange(start, end))
        return np.concatenate(content_indices)
        
    def _get_batch_content_logprobs(batch_distributions: list[Distribution]) -> list[Distribution]:
        with torch.no_grad():
            batch_content_distributions = []

            for distribution in batch_distributions:
                content_indices = _get_content_indices_np(distribution.content_ranges)
                distribution.distribution = distribution.distribution[content_indices]
                content_tokens = distribution.tokenized[content_indices][1:]
                gathered_log_probs = distribution.distribution[np.arange(len(content_indices) - 1), content_tokens]
                distribution.ppl = min(np.exp(-np.mean(gathered_log_probs)), 100)
                batch_content_distributions.append(distribution)

            return batch_content_distributions

    def _process_inference_results(batch_logprobs_np: ndarray, batch_distributions: list[Distribution]) -> int:
        with torch.no_grad():

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

            batch_content_distributions = _get_batch_content_logprobs(batch_distributions)
            disk_queue.put(batch_content_distributions)
            return len(batch_content_distributions)

    exit_flag = False

    while True:
        made_distributions.wait()
        while not result_queue.empty():
            batch_logprobs_np, batch_distributions = result_queue.get()
                
            if batch_distributions is None:
                exit_flag = True
                done_chunk_writes.set()
                break

            batch_distributions_len = _process_inference_results(batch_logprobs_np, batch_distributions)
            pbar_queue.put(("increment", batch_distributions_len))
        
        made_distributions.clear()
        if exit_flag:
            break

class TeacherModel(BaseModel):
    def __init__(self, model_path: str, max_queue_size=3):
        super().__init__(model_path)
        self.distr_device: str = ""
        self.encourage_eos: bool = False
        self.stop_id: int = 0
        self.reserve_vram = []
        self.max_queue_size = max_queue_size

    def process_chunk(self, reserve_vram_gb: list[float] = [], next_stop_id: int = 1, data_manager = None, full_collect: bool = False, validation: bool = False):
        with multiprocessing.Manager() as manager:
            if full_collect:
                num_to_process = self.validation_dataset_len if validation else self.dataset_len
                self.stop_id = self.validation_dataset_len if validation else self.dataset_len
            else:
                num_to_process = self.stop_id - next_stop_id
                self.stop_id = next_stop_id

            done_chunk_writes = manager.Event()
            made_tokens = manager.Event()
            made_distributions = manager.Event()
            inference_queue = manager.Queue(self.max_queue_size)
            result_queue = manager.Queue(self.max_queue_size)
            disk_queue = manager.Queue(self.max_queue_size)
            pbar_queue = manager.Queue(self.max_queue_size)
            dataset = self.validation_dataset if validation else self.dataset

            self.progress_bar = tqdm(total=num_to_process, desc=f"Convos", smoothing=0.06, leave=False)
            self.dataset_len = len(self.dataset)
            self.data_manager = data_manager
            self.reserve_vram = reserve_vram_gb
            
            batch_creator, inference_worker, result_processor = self._start_workers(done_chunk_writes, made_tokens, made_distributions, inference_queue, result_queue, disk_queue, pbar_queue, dataset)

            while not done_chunk_writes.is_set():
                while not disk_queue.empty():
                    batch_content_distributions = disk_queue.get()
                    self.data_manager.write_batch(batch_content_distributions)
                    self.progress_bar.update(len(batch_content_distributions))
                    while not pbar_queue.empty():
                        action, value = pbar_queue.get()
                        self._pbar_actions(action, value)

                while not pbar_queue.empty():
                    action, value = pbar_queue.get()
                    self._pbar_actions(action, value)
                
            self._stop_workers(batch_creator, inference_worker, result_processor)

            self.progress_bar.close()

            self.convo_id = self.stop_id-1

    def sort_dataset_by_len(self):
        self.dataset.sort(key=lambda convo: convo.length, reverse=True)
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
    
    def _start_workers(self, done_chunk_writes, made_tokens, made_distributions, inference_queue, result_queue, disk_queue, pbar_queue, dataset: list[ConvoTokenized]):
        self.progress_bar.set_postfix_str(f"Starting workers for {self.model_name}...")

        batch_creator = multiprocessing.Process(target=_batch_creator_worker, args=(made_tokens, inference_queue, self.batch_size, self.stop_id, dataset))
        inference_worker = multiprocessing.Process(target=_inference_worker, args=(inference_queue, result_queue, made_tokens, made_distributions, self.model_path, self.model_name, self.reserve_vram,
                                                                                         self.device, self.temperature, self.crop_to_size, pbar_queue, self.context_len, self.context_chunk_size, self.batch_size))
        result_processor = multiprocessing.Process(target=_result_processor_worker, args=(result_queue, made_distributions, done_chunk_writes, disk_queue, self.encourage_eos, self.special_tokens, pbar_queue))

        batch_creator.start()
        inference_worker.start()
        result_processor.start()

        return batch_creator, inference_worker, result_processor
    
    def _stop_workers(self, batch_creator: multiprocessing.Process, inference_worker: multiprocessing.Process, result_processor: multiprocessing.Process):
        self.progress_bar.set_postfix_str(f"Stopping workers for {self.model_name}...")
        batch_creator.join()
        inference_worker.join()
        result_processor.join()
    
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