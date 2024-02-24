import torch
import numpy as np
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
from numpy import ndarray
from typing import Optional
import gc
import os
from dataset_utils import get_vocab_family, get_special_tokens
from convert_to_safetensor import convert_model
import json

def is_model_safetensors(model_path: str):
    if os.path.isdir(model_path):
        for file in os.listdir(model_path):
            if file.endswith('.safetensors'):
                return True
        return False
    
def load_prompt_format(prompt_format_path: str) -> None|dict:
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
        'SYS_END': '\n',
        'USER_END': '\n',
        'ASSISTANT_END': '\n'
    }

    keys = list(prompt_format.keys())
    i = 0

    while i < len(keys):
        key = keys[i]
        default_value = prompt_format[key].encode('unicode_escape').decode()
        value = input(f"{key} (default: {default_value}): ")
        
        if value == "<":
            i = max(0, i - 1)
        else:
            prompt_format[key] = value
            i += 1
            
    return prompt_format

def input_config():
    config = {
        'batch_size': 4,
        'add_bos': True,
        'context_chunk_size': 1024
    }
    keys = list(config.keys())
    i = 0

    while i < len(keys):
        key = keys[i]
        default_value = str(config[key])
        value = input(f"{key} (default: {default_value}): ")
        
        if value == "<":
            i = max(0, i - 1)
        else:
            config[key] = value
            i += 1

    return config

class NPDistribution:
    def __init__(self, distribution: ndarray, origin_convo_id: int):
        self.distribution: ndarray = distribution
        self.origin_convo_id: int = origin_convo_id

class ConvoTokenized:
    def __init__(self, tokenized: ndarray, content_ranges, padding, is_empty, cropped_end, convo_id):
        self.tokenized: ndarray = tokenized
        self.content_ranges: list[tuple[int, int]] = content_ranges
        self.padding: int = padding
        self.is_empty: bool = is_empty
        self.cropped_end: bool = cropped_end
        self.length: int = len(tokenized) - padding
        self.len_content: int = sum([end - start for start, end in content_ranges])
        self.origin_convo_id: int = convo_id

class StudentModel:
    def __init__(self, model_path: str):
        self.model_path: str = model_path
        self.model_name: str = ""
        self.model = None
        self.optimizer = None
        self.scheduler = None
        self.device: str = "cuda:0"
        self.prompt_format: dict = {}
        self.batch_size: int = 0
        self.add_bos: bool = False
        self.context_len: int = 0
        self.dataset: list[ConvoTokenized] = []
        self.validation_dataset: list[ConvoTokenized] = []
        self.vocab_family: str = ""
        self.special_tokens: dict = {}
        self.crop_to_size: int = 0
        self.convo_id: int = 0
        self.prepare()

    def prepare(self):
        self.model_name = os.path.basename(self.model_path)
        
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
        self.batch_size = config['batch_size']
        self.add_bos = config['add_bos']
        self.context_chunk_size = config['context_chunk_size']
        self.vocab_family = get_vocab_family(self.model_path)
        self.special_tokens = get_special_tokens(self.model_path)

class TeacherModel:
    def __init__(self, model_path: str):
        self.model_path: str = model_path
        self.model_name: str = ""
        self.model: Optional[ExLlamaV2] = None
        self.device: str = "cuda:0"
        self.distr_device: str = ""
        self.prompt_format: dict = {}
        self.batch_size: int = 0
        self.add_bos: bool = False
        self.context_len: int = 0
        self.context_chunk_size: int = 0
        self.dataset: list[ConvoTokenized] = []
        self.dataset_len: int = 0
        self.dataset_sorted: bool = False
        self.num_batches: int = 0
        self.validation_dataset: list[ConvoTokenized] = []
        self.ppl_dataset: list[ConvoTokenized] = []
        self.vocab_family: str = ""
        self.special_tokens: dict = {}
        self.encourage_eos: bool = False
        self.crop_to_size: int = 0
        self.next_stop_id: int = 0
        self.convo_id: int = 0
        self.prepare()

    def prepare(self):
        self.model_name = os.path.basename(self.model_path)
        
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
        self.batch_size = config['batch_size']
        self.add_bos = config['add_bos']
        self.context_chunk_size = config['context_chunk_size']
        self.vocab_family = get_vocab_family(self.model_path)
        self.special_tokens = get_special_tokens(self.model_path)

    def _inference(self, batch_tokenized: list[np.ndarray]) -> torch.Tensor:
        assert self.model is not None, "Model has not been loaded yet."

        with torch.no_grad():
            batch_tensor = torch.tensor(batch_tokenized, dtype=torch.long, device=self.device)
            batch_logits = self.model.forward(batch_tensor).float() # type: ignore
            if not self.distr_device:
                self.distr_device = str(batch_logits.device) # type: ignore
            return batch_logits # type: ignore
        
    def _get_content_indices_tensor(self, content_ranges, context_len) -> torch.Tensor:
        assert self.distr_device is not "", "Call to get content indices before distribution's device was set."
        content_indices = []
        for start, end in content_ranges:
            if start <= context_len:
                content_indices.append(torch.arange(start, end, device=self.distr_device))
        return torch.cat(content_indices)
    
    def _get_content_indices_np(self, content_ranges, context_len) -> ndarray:
        content_indices = []
        for start, end in content_ranges:
            if start <= context_len:
                content_indices.append(np.arange(start, end))
        return np.concatenate(content_indices)

    def _get_batch_logprobs(self) -> tuple[list[NPDistribution], int]:
        with torch.no_grad():
            batch_tokenized = []
            batch_padding = []
            batch_origins = []

            convos_to_inference = min(self.convo_id + self.batch_size, self.next_stop_id)
            batch_convos = self.dataset[self.convo_id:convos_to_inference]
            batch_size = len(batch_convos)

            for convo in batch_convos:
                batch_tokenized.append(convo.tokenized)
                batch_padding.append(convo.padding)
                batch_origins.append(convo.origin_convo_id)
            
            batch_logits = self._inference(batch_tokenized)
            batch_logprobs = torch.nn.functional.log_softmax(batch_logits, dim=-1)

            batch_distributions_list: list[NPDistribution] = []
            for i in range(0, batch_size):
                convo_logprobs = batch_logprobs[i]
                if self.encourage_eos:
                    eos_id = self.special_tokens['eos']
                    content_ends = [end for start, end in self.dataset[self.convo_id + i].content_ranges]

                    if self.dataset[self.convo_id + i].cropped_end:
                        content_ends = content_ends[:-1]

                    for end in content_ends:
                        convo_logprobs[end][eos_id] = (torch.max(convo_logprobs[end].exp()) * 1.1).log()
                        convo_logprobs[end] = (convo_logprobs[end].exp() / convo_logprobs[end].exp().sum()).log()
                
                np_convo_logprobs = NPDistribution(np.array(batch_logprobs[i].cpu())[:-batch_padding[i]], batch_origins[i])

                batch_distributions_list.append(np_convo_logprobs)
                     
            return batch_distributions_list, batch_size
    
    def get_batch_content_logprobs(self) -> list[NPDistribution]:
        with torch.no_grad():
            batch_distributions, batch_size = self._get_batch_logprobs()
            batch_content_distributions = []
            for i, NPdistribution in enumerate(batch_distributions):
                content_indices = self._get_content_indices_np(self.dataset[self.convo_id + i].content_ranges, self.context_len)
                NPdistribution.distribution = NPdistribution.distribution[content_indices]
                batch_content_distributions.append(NPdistribution)
            
            self.convo_id += batch_size
            
            return batch_content_distributions
        
    def sort_dataset_by_len(self) -> dict[int, int]:
        self.dataset.sort(key=lambda convo: convo.length)
        self.dataset_sorted = True
        sorting_map = {convo.origin_convo_id: index for index, convo in enumerate(self.dataset)}
        return sorting_map
    
    def sort_dataset_by_map(self, sorting_map: dict[int, int]):
        self.dataset.sort(key=lambda convo: sorting_map[convo.origin_convo_id])
        self.dataset_sorted = True
                
    def load_model(self, reserve_vram_gb: list[float] = []):
        print(f"Loading {self.model_name}...")

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
        return model
    
    def unload_model(self):
        assert self.model is not None, "Model has not been loaded yet."
        print(f"Unloading {self.model_name}...")
        self.model.unload()
    
    def unload(self):
        self.unload_model()
        self.model = None
        self.dataset = []
        self.validation_dataset = []
        self.ppl_dataset = []
        self.convo_id = 0
        gc.collect()

    def gen_prelim_ppl(self):
        print(f"Generating preliminary PPL for {self.model_name}...")
        with torch.no_grad():
            total_ppl = 0
            total_tokens = 0
            
                
        
        return total_ppl / total_tokens