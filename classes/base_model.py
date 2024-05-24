from utils.vocab_utils import get_vocab_family, get_special_tokens
from utils.convert_to_safetensor import convert_model
from classes.data_classes import ConvoTokenized
from transformers import AutoTokenizer
from typing import Optional
from tqdm import tqdm
import numpy as np
import codecs
import torch
import json
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
            value = codecs.decode(value, 'unicode_escape')
            prompt_format[key] = value
            i += 1
            
    return prompt_format


def input_config():
    config = {
        'batch_size': 1,
        'add_bos': True,
        'seq_chunk_len': 256,
        'completion': False
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


class BaseModel:
    def __init__(self, model_path: str, student: bool = False):
        self.student: bool = student
        self.model_path: str = model_path
        self.model_name: str = ""
        self.device: str = "cuda:0"
        self.prompt_format: dict = {}
        self.completion: bool = False

        self.batch_size: int = 0
        self.add_bos: bool = False
        self.context_len: int = 0
        self.seq_chunk_len: int = 0

        self.progress_bar: Optional[tqdm] = None

        self.dataset: list[ConvoTokenized] = []
        self.dataset_len: int = 0
        self.dataset_sorted: bool = False

        self.validation_dataset: list[ConvoTokenized] = []
        self.validation_dataset_batched: list[list[ConvoTokenized]] = []
        self.validation_dataset_len: int = 0
        self.validation_dataset_sorted: bool = False

        self.vocab_family: str = ""
        self.special_tokens: dict = {}
        self.temperature: float = 1.0
        self.crop_to_size: int = 0
        self.enable_topK: bool = False
        self.topK: int = 0
        self._prepare()

    def _prepare(self):
        self.model_name = os.path.basename(self.model_path)

        if os.path.exists(f"{self.model_path}_safetensors"):
            self.model_path = f"{self.model_path}_safetensors"

        if not is_model_safetensors(self.model_path):
            self.model_path = convert_model(self.model_path)

        pf = load_prompt_format(self.model_path)
        if pf is None:
            print(f"\n{self.model_name} has no prompt format")
            pf = input_prompt_format()
            save_prompt_format(pf, self.model_path)

        config = load_config(self.model_path)
        if config is None:
            print(f"\n{self.model_name} has no config")
            config = input_config()
            save_config(config, self.model_path)

        self.prompt_format = pf
        self.batch_size = config.get('batch_size', 1)
        self.add_bos = config.get('add_bos', True)
        self.completion = config.get('completion', False)
        
        self.seq_chunk_len = config.get('seq_chunk_len', 256)
        self.vocab_family = get_vocab_family(model_path=self.model_path)
        self.special_tokens = get_special_tokens(model_path=self.model_path)
    
    def write_dataset_to_file(self, folder: str):
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        path = os.path.join(folder, "tokenized_dataset.jsonl")
        path_content = os.path.join(folder, "content_tokenized_dataset.jsonl")
        with open(path, 'w', encoding='utf-8') as file:
            with open(path_content, 'w', encoding='utf-8') as file_content:
                for convo in tqdm(self.dataset, desc="Writing dataset to file"):
                    convo_dict = {
                        "tokenized": convo.tokenized.tolist(),
                        "decoded": [tokenizer.decode(convo.tokenized)],
                        "content_ranges": convo.content_ranges,
                        "content_indices": [list(range(start, end)) for start, end in convo.content_ranges],
                        "content_decoded": [tokenizer.decode(convo.tokenized[start:end]) for start, end in convo.content_ranges],
                        "padding": convo.padding,
                        "cropped_end": convo.cropped_end,
                        "origin_convo_id": convo.origin_convo_id
                    }

                    content_convo_dict = convo_dict.copy()
                    content_convo_dict.pop("tokenized")

                    file.write(json.dumps(convo_dict, ensure_ascii=False) + "\n")
                    file_content.write(json.dumps(content_convo_dict, ensure_ascii=False) + "\n")

