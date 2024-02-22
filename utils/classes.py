import torch
import numpy as np
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
from numpy import ndarray
from typing import Optional
import gc

class ConvoTokenized:
    def __init__(self, tokenized: ndarray, content_ranges, padding, is_empty, cropped_end, convo_id):
        self.tokenized = tokenized
        self.content_ranges = content_ranges
        self.padding = padding
        self.is_empty = is_empty
        self.cropped_end = cropped_end
        self.length = len(tokenized) - padding
        self.len_content = sum([end - start for start, end in content_ranges])
        self.origin_convo_id = convo_id

class StudentModel:
    def __init__(self, model_path: str, model_name: str, prompt_format: dict, config: dict, vocab_family: str, special_tokens: dict):
        self.model_path: str = model_path
        self.model_name: str = model_name
        self.model = None
        self.prompt_format: dict = prompt_format
        self.batch_size: int = config['batch_size']
        self.add_bos: bool = config['add_bos']
        self.dataset: list[ConvoTokenized] = []
        self.validation_dataset: list[ConvoTokenized] = []
        self.vocab_family = vocab_family
        self.special_tokens = special_tokens
        self.crop_to_size: int = 0
        self.convo_id: int = 0

class TeacherModel:
    def __init__(self, model_path: str, model_name: str, prompt_format: dict, config: dict, vocab_family: str, special_tokens: dict, context_len: int):
        self.model_path: str = model_path
        self.model_name: str = model_name
        self.model: Optional[ExLlamaV2] = None
        self.device: str = "cuda:0"
        self.distr_device: str = ""
        self.prompt_format: dict = prompt_format
        self.batch_size: int = config['batch_size']
        self.add_bos: bool = config['add_bos']
        self.context_len: int = context_len
        self.context_chunk_size: int = config['context_chunk_size']
        self.dataset: list[ConvoTokenized] = []
        self.validation_dataset: list[ConvoTokenized] = []
        self.ppl_dataset: list[ConvoTokenized] = []
        self.vocab_family = vocab_family
        self.special_tokens = special_tokens
        self.encourage_eos: bool = False
        self.crop_to_size: int = 0
        self.convo_id: int = 0

    def _inference(self, batch_tokenized: list[np.ndarray]) -> torch.Tensor:
        assert self.model is not None, "Model has not been loaded yet."

        with torch.no_grad():
            batch_tensor = torch.tensor(batch_tokenized, dtype=torch.long).float()
            batch_logits = self.model.forward(batch_tensor)
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

    def _get_batch_logprobs(self) -> list[ndarray]:
        with torch.no_grad():
            batch_tokenized = []
            batch_padding = []

            for i in range(0, self.batch_size):
                convo = self.dataset[self.convo_id + i]
                batch_tokenized.append(convo.tokenized)
                batch_padding.append(convo.padding)
            
            batch_logits = self._inference(batch_tokenized)
            batch_logprobs = torch.nn.functional.log_softmax(batch_logits, dim=-1)

            batch_distributions_list = []
            for i in range(0, self.batch_size):
                convo_logprobs = batch_logprobs[i]
                if self.encourage_eos:
                    eos_id = self.special_tokens['eos']
                    content_ends = [end for start, end in self.dataset[self.convo_id + i].content_ranges]
                    
                    if self.dataset[self.convo_id + i].cropped_end:
                        content_ends = content_ends[:-1]

                    for end in content_ends:
                        convo_logprobs[end][eos_id] = (torch.max(convo_logprobs[end].exp()) * 1.1).log()
                        convo_logprobs[end] = (convo_logprobs[end].exp() / convo_logprobs[end].exp().sum()).log()
                
                np_convo_logprobs = np.array(batch_logprobs[i].cpu())[:-batch_padding[i]]

                batch_distributions_list.append(np_convo_logprobs)
                
            self.convo_id += self.batch_size
            return batch_distributions_list
    
    def get_batch_content_logprobs(self) -> list[ndarray]:
        with torch.no_grad():
            batch_distributions = self._get_batch_logprobs()
            batch_content_distributions = []
            for i in range(0, self.batch_size):
                content_indices = self._get_content_indices_np(self.dataset[self.convo_id + i].content_ranges, self.context_len)
                content_distributions = batch_distributions[i][content_indices]
                batch_content_distributions.append(content_distributions)
            return batch_content_distributions
                
    def load_model(self, reserve_vram_gb: list[float]):
        print(f"Loading {self.model_name}...")

        num_gpus = torch.cuda.device_count()
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
    
    def unload_full(self):
        self.unload_model()
        self.model = None
        self.dataset = []
        self.validation_dataset = []
        self.ppl_dataset = []
        self.convo_id = 0
        gc.collect()
