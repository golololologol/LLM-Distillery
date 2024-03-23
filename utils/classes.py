from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
from transformers import AutoModelForCausalLM, AutoTokenizer
from numpy import ndarray
from typing import Optional
from utils.vocab_utils import get_vocab_family, get_special_tokens
from utils.convert_to_safetensor import convert_model
from utils.dataset_utils import H5DataManager
from tqdm import tqdm
import multiprocessing
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
                # Convert "true", "false" to boolean
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

        self.data_manager: Optional[H5DataManager] = None
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
        self.prepare()

    def prepare(self):
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
    
    def _get_content_indices_np(self, content_ranges) -> ndarray:
        content_indices = []
        for start, end in content_ranges:
            content_indices.append(np.arange(start, end))
        return np.concatenate(content_indices)

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
        self.verbose = True

    def load_model(self):
        if self.verbose:
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
    
    def get_batch_logprobs(self) -> list[Distribution]:
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
        
class TeacherModel(BaseModel):
    def __init__(self, model_path: str, max_queue_size=3):
        super().__init__(model_path)
        self.model: Optional[ExLlamaV2] = None
        self.distr_device: str = ""
        self.encourage_eos: bool = False
        self.stop_id: int = 0
        self.verbose: bool = True
        self.done_chunk_prep = multiprocessing.Event()
        self.done_chunk_writes = multiprocessing.Event()
        self.made_tokens = multiprocessing.Event()
        self.made_distributions = multiprocessing.Event()
        self.inference_queue: multiprocessing.Queue = multiprocessing.Queue(max_queue_size)
        self.result_queue: multiprocessing.Queue = multiprocessing.Queue(max_queue_size)
        self.batch_creator = multiprocessing.Process(target=self._batch_creator_worker)
        self.inference_worker = multiprocessing.Process(target=self._inference_worker)
        self.result_processor = multiprocessing.Process(target=self._result_processor_worker)

    def _batch_creator_worker(self):
        while self.convo_id < self.next_stop_id:
            batch_tokenized, batch_distributions, batch_size = self._prepare_batch_for_inference()
            self.inference_queue.put((batch_tokenized, batch_distributions))
            self.convo_id += batch_size
            self.made_tokens.set()

        self.done_chunk_prep.set()

    def _inference_worker(self):
        while True:
            self.made_tokens.wait()
            while not self.inference_queue.empty():
                batch_tokenized, batch_distributions = self.inference_queue.get()
                batch_logits = self._inference(batch_tokenized)
                self.result_queue.put((batch_logits, batch_distributions))
                self.made_distributions.set()
            self.made_tokens.clear()

    def _result_processor_worker(self):
        while True:
            self.made_distributions.wait()
            while not self.result_queue.empty():
                batch_logits, batch_distributions= self.result_queue.get()
                batch_distributions = self.process_inference_results(batch_logits, batch_distributions)
            self.made_distributions.clear()
            
            if self.done_chunk_prep.is_set():
                self.done_chunk_writes.set()

    def process_chunk(self, next_stop_id: int):
        num_to_process = self.stop_id - next_stop_id
        self.stop_id = next_stop_id
        self.progress_bar = tqdm(total=num_to_process, desc=f"Convos", position=0, smoothing=0.06, leave=False)

        self.load_model()

        

        self._start_workers()

        self.done_chunk_writes.wait()

        self._stop_workers()

    def _inference(self, batch_tokenized: torch.Tensor) -> torch.Tensor:
        assert self.model is not None, f"{self.model_name}: Model has not been loaded yet."

        with torch.no_grad():
            batch_logits = self.model.forward(batch_tokenized).float()/self.temperature # type: ignore

            if not self.distr_device:
                self.distr_device = str(batch_logits.device) # type: ignore
            return batch_logits.to(self.distr_device)[:, :, :self.crop_to_size] # type: ignore

    def _prepare_batch_for_inference(self) -> tuple[torch.Tensor, list[Distribution], int]:
        batch_tokenized: list[np.ndarray] = []
        batch_distributions: list[Distribution] = []

        convos_to_inference = min(self.convo_id + self.batch_size, self.next_stop_id)
        batch_convos = self.dataset[self.convo_id:convos_to_inference]

        for convo in batch_convos:
            batch_distributions.append(Distribution(convo.origin_convo_id, convo.length, convo.cropped_end, convo.content_ranges, convo.tokenized))
            batch_tokenized.append(convo.tokenized)

        # Crop to the convos to one that has the least non-padding tokens
        max_non_padded_len = max(convo.length for convo in batch_convos)
        batch_tokenized = [convo_tokenized[:max_non_padded_len] for convo_tokenized in batch_tokenized]

        batch_tokenized_np = np.array(batch_tokenized)
        batch_tensor = torch.tensor(batch_tokenized_np, dtype=torch.long, device=self.device)
        batch_size = len(batch_distributions)

        return batch_tensor, batch_distributions, batch_size

    def process_inference_results(self, batch_logits: torch.Tensor, batch_distributions: list[Distribution]) -> list[Distribution]:
        with torch.no_grad():
            batch_logprobs = torch.nn.functional.log_softmax(batch_logits, dim=-1)

            for i, distribution in enumerate(batch_distributions):
                convo_logprobs = batch_logprobs[i]
                if self.encourage_eos:
                    eos_id = self.special_tokens['eos_id']
                    content_end_ids = [end-1 for start, end in distribution.content_ranges]

                    if distribution.cropped_end:
                        content_end_ids = content_end_ids[:-1]

                    for end in content_end_ids:
                        convo_logprobs[end][eos_id] = (torch.max(convo_logprobs[end].exp()) * 1.1).log()
                        convo_logprobs[end] = (convo_logprobs[end].exp() / convo_logprobs[end].exp().sum()).log()

                distribution.distribution = convo_logprobs[:distribution.length].cpu().numpy()

            batch_content_distributions = self.get_batch_content_logprobs(batch_distributions)
            H5DataManager.write_batch(batch_content_distributions)

    def get_batch_content_logprobs(self, batch_distributions: list[Distribution]) -> tuple[list[Distribution], int]:
        with torch.no_grad():
            batch_content_distributions = []
            for distribution in batch_distributions:
                content_indices = self._get_content_indices_np(distribution.content_ranges)
                distribution.distribution = distribution.distribution[content_indices]
                content_tokens = distribution.tokenized[content_indices][1:]
                gathered_log_probs = distribution.distribution[np.arange(len(content_indices) - 1), content_tokens]
                distribution.ppl = min(np.exp(-np.mean(gathered_log_probs)), 100)
                batch_content_distributions.append(distribution)

            return batch_content_distributions
        
    def sort_dataset_by_len(self):
        self.dataset.sort(key=lambda convo: convo.length, reverse=True)
        self.dataset_sorted = True
  
    def load_model(self, reserve_vram_gb: list[float] = []):
        if self.verbose:
            print(f" Loading {self.model_name}...")

        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise Exception("No CUDA-capable GPUs found.")
        
        reserve_vram_kb = [int(128 * 1000**2)]*num_gpus # default 128MB/GPU

        for i, reserve_gb in enumerate(reserve_vram_gb):
            if reserve_gb > 0 and i < num_gpus:
                reserve_vram_kb[i] = int(reserve_gb * 1000**3)

        config = ExLlamaV2Config()
        config.model_dir = self.model_path
        config.prepare()
        config.max_seq_len = self.context_len
        config.max_batch_size = self.batch_size
        config.max_input_len = self.context_chunk_size * self.batch_size
        config.max_attention_size = self.context_chunk_size * self.batch_size

        model = ExLlamaV2(config)
        cache = ExLlamaV2Cache(model, lazy=True)

        model.load_autosplit(cache, reserve_vram=reserve_vram_kb)
        self.model = model
    
    def unload_model(self):
        if self.model is None:
            print(f" {self.model_name} is already unloaded.")
            return
        if self.verbose:
            print(f" Unloading {self.model_name}...")
        self.model.unload()
    
    def unload_full(self):
        self.unload_model()
        self.model = None
        self.dataset = []
        self.validation_dataset = []
        self.convo_id = 0
        gc.collect()

    def _start_workers(self):
        self.batch_creator.start()
        self.inference_worker.start()
        self.result_processor.start()
    
    def _stop_workers(self):
        self.batch_creator.terminate()
        self.inference_worker.terminate()
        self.result_processor.terminate()
    
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