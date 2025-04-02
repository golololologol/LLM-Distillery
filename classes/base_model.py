from utils.convert_to_safetensor import convert_model
from classes.data_classes import ConvoTokenized
from utils.vocab_utils import get_vocab_family
from classes.args import PipelineConfig
from transformers import AutoTokenizer
from typing import Optional, Any
from tqdm import tqdm
import codecs
import json
import os


__CONFIG_NAME__ = "pipeline_config.json"
__PROMPT_FORMAT_NAME__ = "prompt_format.json"


def is_model_safetensors(model_path: str):
    if os.path.isdir(model_path):
        for file in os.listdir(model_path):
            if file.endswith('.safetensors'):
                return True
    return False
        

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
    """
    Base class for all models, contains common attributes and methods.
    """
    def __init__(self, pipe_config: PipelineConfig, model_path: str, student: bool = False):
        self.student: bool = student
        self.model_path: str = model_path
        
        self.model_name: str = ""
        self.device: str = pipe_config.device
        self.prompt_format: dict = None
        self.completion: bool = None

        self.batch_size: int = None
        self.add_bos: bool = None
        self.context_len: int = None
        self.seq_chunk_len: int = None

        self.progress_bar: Optional[tqdm] = None

        self.dataset: list[ConvoTokenized] = None
        self.dataset_len: int = None
        self.dataset_sorted: bool = None

        self.validation_dataset: list[ConvoTokenized] = None
        self.validation_dataset_batched: list[list[ConvoTokenized]] = None
        self.validation_dataset_len: int = None
        self.validation_dataset_sorted: bool = None
        
        self.vocab: dict = {}
        self.vocab_tokens_set: set = set()
        self.vocab_family: str = ""
        self.special_tokens: dict = {}
        self.temperature: float = pipe_config.temperature
        self.crop_to_size: int = pipe_config.crop_distr_to_size
        self.enable_topK: bool = pipe_config.enable_topK
        self.topK: int = pipe_config.save_topK
        self._prepare(pipe_config)

    def _prepare(self, pipe_config: PipelineConfig):
        self.model_name = os.path.basename(self.model_path)

        if os.path.exists(f"{self.model_path}_safetensors"):
            self.model_path = f"{self.model_path}_safetensors"

        if not is_model_safetensors(self.model_path):
            self.model_path = convert_model(self.model_path)

        if self.student:
            pf = pipe_config.prompt_format
            config = {
                'batch_size': pipe_config.batch_size,
                'add_bos': pipe_config.add_bos,
            }
        else:
            pf = self.load_json_data(__PROMPT_FORMAT_NAME__)
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
        self.vocab_family = get_vocab_family(model_path=self.model_path)
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        self.vocab = tokenizer.get_vocab()
        self.vocab_tokens_set = set(self.vocab.keys())
    
    def _write_convo_to_file(self, file, file_content, convo, tokenizer):
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

    def write_dataset_to_file(self, folder: str, identifier: str = "", validation: bool = False):
        """
        Write the dataset to a file in JSONL format. Each line is a JSON object representing a conversation 

        Args:
            folder (str): The folder where the dataset will be saved.
            identifier (str): An optional identifier to append to the filenames.
            validation (bool): Whether to write the validation dataset or the main dataset.
            
        
        """
        insert = "validation_" if validation else ""
        dataset = self.validation_dataset if validation else self.dataset
        
        tokenizer = AutoTokenizer.from_pretrained(self.model_path)
        path = os.path.join(folder, f"{identifier}_tokenized_{insert}dataset.jsonl")
        path_content = os.path.join(folder, f"{identifier}_content_tokenized_{insert}dataset.jsonl")
        with open(path, 'w', encoding='utf-8') as file:
            with open(path_content, 'w', encoding='utf-8') as file_content:
                for convo in tqdm(dataset, desc="Writing dataset to file"):
                    self._write_convo_to_file(file, file_content, convo, tokenizer)
                    
    def load_json_data(self, file_name: str) -> None | Any:
        """
        Load JSON data from a file in the model directory.\\
        If the file does not exist, return None. Else return the data.
        
        Args:
            file_name (str): The name of the file to load.

        Returns:
            None | Any: The loaded data or None if the file does not exist.
        """
        
        data_path = os.path.join(self.model_path, file_name)
        
        if not os.path.exists(data_path):
            return None
    
        with open(data_path, 'r', encoding='utf-8') as file:
            return json.load(file)
        
    def save_json_data(self, data: Any, file_name: str) -> None:
        """
        Save JSON data to a file in the model directory.\\
        If the file does not exist, create it.\\
        If the file exists, overwrite it.

        Args:
            data (Any): The data to save.
            file_name (str): The name of the file to save the data to.
        """
        
        data_path = os.path.join(self.model_path, file_name)
        
        with open(data_path, 'w', encoding='utf-8') as file:
            json.dump(data, file, ensure_ascii=False, indent=4)
