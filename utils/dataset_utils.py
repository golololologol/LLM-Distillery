import os
import numpy as np
import h5py
import threading
import queue
import json
import torch
from transformers import AutoTokenizer

class H5Writer:
    def __init__(self, save_folder: str):
        self.save_path = os.path.join(save_folder, "distributions.hdf5")
        if os.path.exists(self.save_path):
            os.remove(self.save_path)
        self.queue = queue.Queue(maxsize=2)
        self.thread = threading.Thread(target=self._save_to_hdf5_thread)
        self.thread.start()
        
    def _save_to_hdf5_thread(self):
        with h5py.File(self.save_path, 'a') as hdf_file:
            convo_count = 0
            while True:
                item = self.queue.get()
                if item is None:
                    break
                dataset_name = f'convo_{convo_count}'
                hdf_file.create_dataset(dataset_name, data=item)
                convo_count += 1
                self.queue.task_done()
            
    def write_data(self, data: torch.Tensor):
        self.queue.put(data.to('cpu'))

    def close(self):
        self.queue.put(None)
        self.thread.join()

class H5Reader:
    def __init__(self, model_path, device, empty_convo_ids=[], queue_size=2, timeout=15):
        self.file_path = os.path.join(model_path, "distributions.hdf5")
        self.device = device
        self.timeout = timeout
        self.empty_convo_ids = empty_convo_ids
        self.queue = queue.Queue(maxsize=queue_size)
        self.stop_loading = threading.Event()
        self.loading_thread = threading.Thread(target=self._load_data)
        self.loading_thread.start()

    def _load_data(self):
        with h5py.File(self.file_path, 'r') as hdf_file:
            while True:
                dataset_names = sorted(hdf_file.keys(), key=lambda x: int(x.split('_')[1]))
                for i, dataset_name in enumerate(dataset_names):
                    if self.stop_loading.is_set():
                        return
                    if i in self.empty_convo_ids:
                        continue
                    tensor = torch.tensor(np.array(hdf_file[dataset_name]), device=self.device)
                    self.queue.put(tensor, timeout=self.timeout)

    def read_next(self) -> torch.Tensor:
        tensor = self.queue.get()
        self.queue.task_done()
        return tensor

    def close(self):
        self.stop_loading.set()
        self.queue.get()
        self.queue.task_done()
        self.loading_thread.join()

def read_jsonl_lazy(file_path): # Generator to lazy read the dataset line-by-line
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                data = json.loads(line)
                yield data

def good_encode(text: str, encode_special = True, replace_tokens = True, tokenizer=None, model_path="") -> torch.Tensor:
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    if replace_tokens:
        text = text.replace('<bos>', tokenizer.bos_token).replace('<eos>', tokenizer.eos_token)
    return tokenizer.encode("\n" + text, add_special_tokens=False, return_tensors="pt").squeeze(0)[2:] # type: ignore

def encode_prompt_format(prompt_format: dict, tokenizer=None, model_path="") -> dict[str, torch.Tensor]:
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    for key, value in prompt_format.items():
        prompt_format[key] = good_encode(value, tokenizer=tokenizer)
    return prompt_format

def tokenize_dataset(dataset_path, device, sort, model_path, prompt_format, context_len, save_sys_range, save_user_range, save_assistant_range):
    print("Tokenizing the dataset...")

    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    pf = encode_prompt_format(prompt_format, tokenizer)

    total_tokens = 0
    empty_convo_ids = []
    dataset_tokenized = []
    dataset_content_ranges = []

    for convo_id, item in enumerate(read_jsonl_lazy(dataset_path)):  # Every conversation
        conversation_tokenized = torch.Tensor().to(device)
        conversation_content_ranges = []
        start_index = 0
        num_turns = len(item["conversations"])

        if num_turns < 2:
            empty_convo_ids.append(convo_id)
            dataset_tokenized.append(conversation_tokenized)
            dataset_content_ranges.append(conversation_content_ranges)
            continue

        tags = item.get("tags", [])
        reversed = ("reversed" in tags) or item.get("reversed", False)

        sys_content = item.get("init", "")
        if sys_content:
            sys_content_tokenized = good_encode(sys_content.strip(), replace_tokens=False, encode_special=False, tokenizer=tokenizer)
            sys_tokenized = torch.cat((pf['SYS_START'], sys_content_tokenized, pf['SYS_END']))
            conversation_tokenized = sys_tokenized
            if save_sys_range:
                conversation_content_ranges.append((pf['SYS_START'].numel() - 1, sys_tokenized.numel() - pf['SYS_END'].numel()))
            start_index = sys_tokenized.numel()

        for i, turn in enumerate(item["conversations"]):  # Every turn
            if reversed:
                assistant = (i + 1) % 2
            else:
                assistant = i % 2

            turn_start_tokenized = pf['ASSISTANT_START'] if assistant else pf['USER_START']
            turn_end_tokenized = pf['ASSISTANT_END'] if assistant else pf['USER_END']
            turn_tokenized = good_encode(turn.strip(), replace_tokens=False, encode_special=False, tokenizer=tokenizer)

            full_turn_tokenized = torch.cat((turn_start_tokenized, turn_tokenized, turn_end_tokenized))
            end_index = start_index + full_turn_tokenized.numel()

            if save_assistant_range and assistant:
                content_start_index = start_index + pf['ASSISTANT_START'].numel() - 1
                content_end_index = end_index - pf['ASSISTANT_END'].numel()
                conversation_content_ranges.append((content_start_index, content_end_index))
            elif save_user_range and not assistant:
                content_start_index = start_index + pf['USER_START'].numel() - 1
                content_end_index = end_index - pf['USER_END'].numel()
                conversation_content_ranges.append((content_start_index, content_end_index))

            start_index = end_index

            conversation_tokenized = torch.cat((conversation_tokenized, full_turn_tokenized)) if conversation_tokenized.numel() > 0 else full_turn_tokenized

        if conversation_content_ranges[0][0] > context_len:
            empty_convo_ids.append(convo_id)

        total_tokens += conversation_tokenized.numel()
        dataset_tokenized.append(conversation_tokenized.to("cpu"))
        if reversed:
            del conversation_content_ranges[0]
        dataset_content_ranges.append(conversation_content_ranges)

    if sort:
        combined_list = list(zip(dataset_tokenized, dataset_content_ranges))
        combined_list.sort(key=lambda x: x[0].shape[0], reverse=True)
        dataset_tokenized, dataset_content_ranges = map(list, zip(*combined_list))

    print(f"Total processed tokens: {total_tokens}")
    print(f"Total content tokens saved: {sum([range[1] - range[0] for ranges in dataset_content_ranges for range in ranges])}")
    return dataset_tokenized, dataset_content_ranges, empty_convo_ids

def generate_metadata(model_path: str, dataset_tokenized: list, dataset_content_ranges: list, empty_convo_ids=[]) -> dict:
    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)

    if not (dataset_tokenized and dataset_content_ranges):
        first_convo_decoded = ""
        first_content_decoded = []
    else:
        first_convo_decoded = tokenizer.decode(dataset_tokenized[0])
        first_content_decoded = []
        for content_range in dataset_content_ranges[0]:
            content_start, content_end = content_range
            content_decoded = tokenizer.decode(dataset_tokenized[0][content_start:content_end])
            first_content_decoded.append(content_decoded)

    special_tokens = {tokenizer.unk_token, tokenizer.bos_token, tokenizer.eos_token, tokenizer.pad_token}
    all_added_tokens = tokenizer.get_added_vocab()

    added_tokens_ids = [v for k, v in all_added_tokens.items() if k not in special_tokens]

    vocab = {k: v for k, v in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])}
    
    metadata = {
            "first_convo_decoded": first_convo_decoded,
            "first_content_decoded": first_content_decoded,
            "dataset_len": len(dataset_tokenized),
            "total_tokens": sum([convo_tokenized.numel() for convo_tokenized in dataset_tokenized]),
            "total_content_tokens": sum([sum([content_range[1] - content_range[0] for content_range in convo_content_ranges]) for convo_content_ranges in dataset_content_ranges]),
            "empty_convo_ids": empty_convo_ids,
            "merged": False,
            "merged_models": [],
            "model_name": os.path.basename(os.path.normpath(model_path)),
            "bos_id": tokenizer.bos_token_id,
            "eos_id": tokenizer.eos_token_id,
            "pad_id": tokenizer.pad_token_id,
            "unk_id": tokenizer.unk_token_id,
            "added_tokens_ids": added_tokens_ids,
            "vocab_family": get_vocab_family(tokenizer),
            "vocab": vocab
            }
    
    return metadata

def get_vocab_family(tokenizer) -> str:
    vocab_size_to_family = {
        30000: "GPT2",
        32000: {
            "监": "Mistral",
            "Z": "LLAMA"
        },
        50265: "GPTNeo",
        50257: "GPTJ"
    }
    vocab_family = vocab_size_to_family.get(tokenizer.vocab_size, "Unknown")
    if vocab_family == "Unknown":
        token_29999 = tokenizer.convert_ids_to_tokens(29999)
        if token_29999 == "监":
            vocab_family = "Mistral"
        elif token_29999 == "Z":
            vocab_family = "LLAMA"
            
    if isinstance(vocab_family, dict):
        token_29999 = tokenizer.convert_ids_to_tokens(29999)
        vocab_family = vocab_family.get(token_29999, "Unknown")
    return vocab_family

def save_dataset_and_metadata(dataset_tokenized: list, dataset_content_ranges: list, metadata: dict, save_folder: str):
    save_tokenized_dataset(dataset_tokenized, dataset_content_ranges, save_folder)
    save_metadata(metadata, save_folder)

def save_tokenized_dataset(dataset_tokenized: list, dataset_content_ranges: list, save_folder: str):
    file = os.path.join(save_folder, "dataset_tokenized.jsonl")

    with open(file, 'w', encoding='utf-8') as f:
        for i, convo_tokenized in enumerate(dataset_tokenized):
            content_tokens = []
            for content_range in dataset_content_ranges[i]:
                content_start, content_end = content_range
                content_tokens.extend(convo_tokenized[content_start:content_end].tolist())

            data_to_save = {
                "convo_tokenized": convo_tokenized.tolist(),
                "content_ranges": dataset_content_ranges[i],
                "content_tokens": content_tokens
            }
            f.write(json.dumps(data_to_save, ensure_ascii=False) + '\n')

def save_sorted_dataset(save_folder: str, dataset_path: str):
    file = os.path.join(save_folder, "dataset_sorted.jsonl")
    with open(file, 'w', encoding='utf-8') as f:
        with open(dataset_path, 'r', encoding='utf-8') as df:
            data = []
            for line in df:
                data.append(json.loads(line))
            data.sort(key=lambda x: sum(len(turn) for turn in x["conversations"]), reverse=True)
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + '\n')

def filter_empty_conversations(dataset_tokenized, dataset_content_ranges, empty_convo_ids):
    dataset_tokenized_filtered = []
    dataset_content_ranges_filtered = []
    for i, convo_tokenized in enumerate(dataset_tokenized):
        if i not in empty_convo_ids:
            dataset_tokenized_filtered.append(convo_tokenized)
            dataset_content_ranges_filtered.append(dataset_content_ranges[i])
    return dataset_tokenized_filtered, dataset_content_ranges_filtered

def save_metadata(metadata: dict, save_folder: str):
    metadata_file = os.path.join(save_folder, "dataset_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as meta_f:
        json.dump(metadata, meta_f, ensure_ascii=False, indent=4)

def load_metadata(distributions_path: str) -> dict:
    metadata_path = os.path.join(distributions_path, "dataset_metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as file:
        metadata = json.load(file)
        return metadata
    