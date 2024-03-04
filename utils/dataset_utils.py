import os
import numpy as np
import h5py
import time
import threading
import queue
import json
import torch
import logging
from tqdm import tqdm
from transformers import AutoTokenizer
from exllamav2 import ExLlamaV2Tokenizer, ExLlamaV2Config
from classes import ConvoTokenized, Distribution
logging.getLogger("transformers").setLevel(logging.ERROR)  # Shut up transformers
logging.getLogger("torch").setLevel(logging.ERROR) # And pytorch for good measure

class H5Writer:
    def __init__(self, save_folder: str, timeout=15, queue_size=2):
        self.save_path = os.path.join(save_folder, "distributions.hdf5")
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
        self.timeout = timeout
        self.queue = queue.Queue(maxsize=queue_size)
        self.thread = threading.Thread(target=self._save_to_hdf5_thread)
        self.thread.start()
        
    def _save_to_hdf5_thread(self):
        with h5py.File(self.save_path, 'a') as hdf_file:
            convo_count = 0
            while True:
                item = self.queue.get(timeout=self.timeout)
                if item is None:
                    break
                origin_convo_id = item.origin_convo_id
                dataset_name = f'convo_{origin_convo_id}'

                hdf_file.create_dataset(dataset_name, data=item.distribution.astype(np.float16))
                convo_count += 1
                self.queue.task_done()
            
    def write_data(self, data: Distribution):
        self.queue.put(data)

    def write_batch(self, batch: list[Distribution]):
        for item in batch:
            self.queue.put(item)

    def close(self):
        self.queue.put(None)
        self.thread.join()

class H5Reader:
    def __init__(self, model_path, device, empty_convo_ids=[], queue_size=2, timeout=15, shuffle=False):
        self.file_path = os.path.join(model_path, "distributions.hdf5")
        self.device = device
        self.timeout = timeout
        self.empty_convo_ids = empty_convo_ids
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_loading = threading.Event()
        self.order_list = []
        self.dataset_len = 0
        self.awaiting = False
        self.target = None
        if shuffle:
            target = self._load_data_shuffled
        else:
            target = self._load_data
        self.loading_thread = threading.Thread(target=target)
        self.loading_thread.start()

    def restart_thread(self):
        self.stop_loading.set()
        self.loading_thread.join()
        self.stop_loading.clear()
        self.queue = queue.Queue(maxsize=self.queue.maxsize)
        self.loading_thread = threading.Thread(target=self.target)
        self.loading_thread.start()

    def _load_data(self):
        with h5py.File(self.file_path, 'r') as hdf_file:
            self.dataset_len = len(hdf_file)
            while True:
                dataset_names = sorted(hdf_file.keys(), key=lambda x: int(x.split('_')[1]))
                for id, dataset_name in enumerate(dataset_names):
                    if id in self.empty_convo_ids:
                        continue

                    self._await()

                    np_distribution = Distribution(np.array(hdf_file[dataset_name]).astype(np.float32), id)
                    
                    self.queue.put(np_distribution, block=False)

    def _load_data_shuffled(self):
        with h5py.File(self.file_path, 'r') as hdf_file:
            while True:
                self._await_order_list()
                for id in self.order_list:
                    if id in self.empty_convo_ids:
                        continue
                    self._await()
                    dataset_name = f'convo_{id}'
                    np_distribution = Distribution(np.array(hdf_file[dataset_name]).astype(np.float32), id)
                    self.queue.put(np_distribution, block=False)
                self.order_list = []

    def _await(self):
        time_left = self.timeout
        while self.queue.qsize() >= self.queue.maxsize - 1:
            if self.stop_loading.is_set():
                return
            time.sleep(0.05)
            if not self.awaiting:
                time_left -= 0.05
            if time_left <= 0:
                return
            
    def _await_order_list(self):
        while not self.order_list:
            if self.stop_loading.is_set():
                return
            time.sleep(0.05)

    def toggle_await(self):
        self.awaiting = not self.awaiting

    def read_next(self) -> torch.Tensor:
        tensor = self.queue.get()
        self.queue.task_done()
        return tensor

    def set_loading_order(self, order_list):
        self.order_list = order_list

    def close(self):
        self.stop_loading.set()
        self.loading_thread.join()

def read_jsonl_lazy(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        line_number = 0
        for line in f:
            line_number += 1
            try:
                data = json.loads(line)
                yield data
            except json.JSONDecodeError as e:
                print(f"Fuck up on line {line_number}: {e.msg}. Line content: {line.strip()}")

def get_special_tokens(vocab_family) -> dict[str, str|int]:
    if vocab_family == "LLAMA":
        sp_toks = {
            "bos": "<s>",
            "bos_id": 1,
            "eos": "</s>",
            "eos_id": 2,
            "pad": None,
            "pad_id": None,
            "unk": "<unk>",
            "unk_id": 0
        }
    else:
        raise NotImplementedError(f"{vocab_family} not yet supported")
    return sp_toks

def good_encode(text: str, sp_toks: dict, tokenizer, encode_special=True, replace_tokens=True):
    if replace_tokens:
        text = text.replace('<bos>', sp_toks["bos"]).replace('<eos>', sp_toks["eos"])

    if tokenizer.__class__.__name__ != "ExLlamaV2Tokenizer":
        encoded_ids = tokenizer.encode("\n" + text, add_special_tokens=False)[2:]
    else:
        encoded_ids = tokenizer.encode("\n" + text, encode_special_tokens=encode_special).squeeze(0)[2:] # type: ignore

    encoded_text = np.array(encoded_ids, dtype=np.int64)

    return encoded_text

def encode_prompt_format(prompt_format: dict, sp_toks: dict, tokenizer) -> dict[str, list[int]]:
    prompt_format = prompt_format.copy()

    for key, value in prompt_format.items():
        prompt_format[key] = good_encode(value, sp_toks, tokenizer)
    return prompt_format

def tokenize_convo(json_item, sp_toks, tokenizer, pf, save_sys_range, save_user_range, save_assistant_range, context_len, add_bos=True):
    empty = False

    if add_bos:
        conversation_tokenized = np.array([sp_toks["bos_id"]], dtype=np.int64)
        start_index = 0
    else:
        conversation_tokenized = np.array([], dtype=np.int64)
        start_index = -1

    conversation_content_ranges = []
    num_turns = len(json_item["conversations"])

    if num_turns < 2:
        empty = True
        return conversation_tokenized, conversation_content_ranges, empty

    tags = json_item.get("tags", [])
    reversed = ("reversed" in tags) or json_item.get("reversed", False)

    sys_content = json_item.get("init", "")
    if sys_content:
        sys_content_tokenized = good_encode(sys_content.strip(), sp_toks, tokenizer, replace_tokens=False, encode_special=False)
        sys_tokenized = np.concatenate((conversation_tokenized, pf['SYS_START'], sys_content_tokenized, pf['SYS_END']))
        if save_sys_range:
            conversation_content_ranges.append((len(pf['SYS_START']) + start_index, len(sys_tokenized) - len(pf['SYS_END'])))
        start_index = len(sys_tokenized) - 1

    for i, turn in enumerate(json_item["conversations"]):
        if reversed:
            assistant = (i + 1) % 2
        else:
            assistant = i % 2

        turn_start_tokenized = pf['ASSISTANT_START'] if assistant else pf['USER_START']
        turn_end_tokenized = pf['ASSISTANT_END'] if assistant else pf['USER_END']
        turn_tokenized = good_encode(turn.strip(), sp_toks, tokenizer, replace_tokens=False, encode_special=False)
        turn_len = len(turn_tokenized)

        full_turn_tokenized = np.concatenate((turn_start_tokenized, turn_tokenized, turn_end_tokenized))
        end_index = start_index + len(full_turn_tokenized)

        if save_assistant_range and assistant:
            content_start_index = start_index + len(pf['ASSISTANT_START'])
            content_end_index = content_start_index + turn_len + 1
            conversation_content_ranges.append((content_start_index, content_end_index))
        if save_user_range and not assistant:
            content_start_index = start_index + len(pf['USER_START'])
            content_end_index = content_start_index + turn_len + 1
            conversation_content_ranges.append((content_start_index, content_end_index))

        start_index = end_index

        if len(conversation_tokenized) > 0:
            conversation_tokenized = np.concatenate((conversation_tokenized, full_turn_tokenized))
        else:
            conversation_tokenized = full_turn_tokenized
        
    if conversation_content_ranges[0][0] > context_len:
        empty = True

    return conversation_tokenized, conversation_content_ranges, empty

def tokenize_dataset(dataset_path, model_path, prompt_format, context_len, save_sys_range, save_user_range, save_assistant_range, add_bos=True) -> list[ConvoTokenized]:
    try:
        config = ExLlamaV2Config()
        config.model_dir = model_path
        config.prepare()
        tokenizer = ExLlamaV2Tokenizer(config)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)

    vocab_family = get_vocab_family(model_path=model_path)
    sp_toks = get_special_tokens(vocab_family)
    pf = encode_prompt_format(prompt_format, sp_toks, tokenizer)
    
    dataset = []

    for convo_id, item in tqdm(enumerate(read_jsonl_lazy(dataset_path)), desc="Tokenizing", unit="convo"): # Every conversation
        conversation_tokenized, conversation_content_ranges, empty = tokenize_convo(item, sp_toks, tokenizer, pf, save_sys_range, save_user_range, save_assistant_range, context_len, add_bos=add_bos)
        
        num_pad_tokens = 0
        cropped_end = False

        if len(conversation_tokenized) > context_len:
            conversation_tokenized = conversation_tokenized[:context_len]
        else:
            num_pad_tokens = context_len - len(conversation_tokenized)
            conversation_tokenized = np.concatenate((conversation_tokenized, np.array([sp_toks["eos_id"]] * num_pad_tokens, dtype=np.int64)))

        for start, end in conversation_content_ranges:
            corrected_content_ranges = []
            if start > context_len:
                continue
            if end > context_len:
                end = context_len
                cropped_end = True
            corrected_content_ranges.append((start, end))

        conversation = ConvoTokenized(conversation_tokenized, corrected_content_ranges, num_pad_tokens, empty, cropped_end, convo_id)
        dataset.append(conversation)

    return dataset

def get_vocab_family(tokenizer=None, model_path="") -> str:
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)

    token_29999_to_family = {
        "监": "Mistral",
        "Z": "LLAMA"
    }

    token_29999 = tokenizer.convert_ids_to_tokens(29999)
    vocab_family = token_29999_to_family.get(token_29999, "Unknown") # type: ignore
    return vocab_family
    