import os
import numpy as np
import h5py
import multiprocessing
import json
import logging
import time
from tqdm import tqdm
from transformers import AutoTokenizer
from exllamav2 import ExLlamaV2Tokenizer, ExLlamaV2Config
from utils.classes import ConvoTokenized, Distribution
from multiprocessing import shared_memory
from utils.vocab_utils import get_special_tokens
logging.getLogger("transformers").setLevel(logging.ERROR)  # Shut up transformers
logging.getLogger("torch").setLevel(logging.ERROR) # And pytorch for good measure

class H5DataManager:
    def __init__(self, dataset_path, device, max_queue_size=3, done_everything=multiprocessing.Event()):
        self.file_path = os.path.join(dataset_path, "distributions.hdf5")
        self.device = device
        self.queue = multiprocessing.Queue(max_queue_size)
        self.result_queue = multiprocessing.Queue(max_queue_size)
        self.done_everything = done_everything
        self.got_task = multiprocessing.Event()
        self.loading_process = multiprocessing.Process(target=self._process_thread)
        self.teacher_convos = []
        self.shared_batches = []

        self.loading_process.start()

    def _process_thread(self):
        with h5py.File(self.file_path, 'a') as hdf_file:
            while True:
                self.got_task.wait()
                self.done_everything.clear()
                self.shared_batches = []
                
                while not self.queue.empty():
                    task, data = self.queue.get()
                    match task:
                        case 'get_batch':
                            self.result_queue.put(self._make_outgoing_batch(hdf_file, data))
                        case 'get_batches':
                            for batch_ids in data:
                                self.result_queue.put(self._make_outgoing_batch(hdf_file, batch_ids))
                        case 'put_batch':
                            self._process_distributions(hdf_file, data)
                        case 'get_available_ids':
                            self.result_queue.put([int(dataset_name.split('_')[1]) for dataset_name in hdf_file])
                        case 'clear_dataset':
                            self._clear_dataset(hdf_file)
                        case 'exit':
                            self.got_task.clear()
                            return
                        
                self.done_everything.set()
                self.got_task.clear()

    def _process_distributions(self, hdf_file, batch: list[Distribution]):
        for distribution in batch:

            distr_shd_mem = shared_memory.SharedMemory(name=distribution.shd_mem_name)
            distribution.distribution = np.ndarray(distribution.distr_shape, dtype=distribution.distr_dtype, buffer=distr_shd_mem.buf)

            id = distribution.origin_convo_id
            if id in self.teacher_convos:
                merged_data = self._merge_data(hdf_file, distribution, id)
                self._save_data(hdf_file, merged_data, id)
            else:
                self._save_data(hdf_file, distribution.distribution, id)
                self.teacher_convos.append(id)
            distr_shd_mem.close()
            distr_shd_mem.unlink()

    def _merge_data(self, hdf_file, new_distribution: Distribution, id):
        assert isinstance(new_distribution.distribution, np.ndarray), "Distribution should be a numpy array."
        
        ppl = new_distribution.ppl
        assert ppl > 0, "PPL should be greater than 0."

        disk_data = np.exp(self._load_id(hdf_file, id))
        new_data = np.exp(new_distribution.distribution) / np.log(ppl)

        raw_diff = disk_data.shape[0] - new_data.shape[0]

        if raw_diff < 0:
            disk_data = np.pad(disk_data, ((0, -raw_diff), (0, 0)))
        elif raw_diff > 0:
            new_data = np.pad(new_data, ((0, raw_diff), (0, 0)))

        merged_data = disk_data + new_data

        return np.log(merged_data)

    def _save_data(self, hdf_file: h5py.File, data: np.ndarray, convo_id: int):
        dataset_name = f'convo_{convo_id}'
        if dataset_name in hdf_file:
            del hdf_file[dataset_name]
        hdf_file.create_dataset(dataset_name, data=data.astype(np.float16))
    
    def _load_id(self, hdf_file, convo_id: int) -> np.ndarray:
        data = np.array(hdf_file[f'convo_{convo_id}']).astype(np.float32) if f'convo_{convo_id}' in hdf_file else None
        return data
    
    def _make_outgoing_batch(self, hdf_file, batch_ids: list[int]) -> tuple[str, tuple[int, int], np.dtype]:
        batch = [self._load_id(hdf_file, convo_id) for convo_id in batch_ids]

        max_len = max([len(distr) for distr in batch])
        padded_batch = np.array([np.pad(distr, ((0, max_len - len(distr)), (0, 0))) for distr in batch])

        shared_batch_memory = shared_memory.SharedMemory(create=True, size=padded_batch.nbytes)
        shared_batch = np.ndarray(padded_batch.shape, dtype=padded_batch.dtype, buffer=shared_batch_memory.buf)
        np.copyto(shared_batch, padded_batch)
        self.shared_batches.append(shared_batch_memory)

        return shared_batch_memory.name, shared_batch.shape, shared_batch.dtype

    def _clear_dataset(self, hdf_file: h5py.File):
        for dataset_name in hdf_file:
            del hdf_file[dataset_name]
        self.teacher_convos = []
        self.result_queue.put(True)
    
    def get_batch(self, convo_ids: list[int]) -> np.ndarray:
        self.queue.put(('get_batch', convo_ids))
        self.got_task.set()
        return self.result_queue.get()
    
    def enqueue_get_batches(self, batches: list[list[int]]):
        self.queue.put(('get_batches', batches))
        self.got_task.set()
    
    def read_next_batch(self) -> np.ndarray:
        return self.result_queue.get()

    def write_batch(self, batch: list[Distribution]):
        self.queue.put(('put_batch', batch))
        self.got_task.set()

    def get_dataset_ids(self) -> list[int]:
        self.queue.put(('get_available_ids', None))
        self.got_task.set()
        return self.result_queue.get()

    def purge_dataset(self):
        self.queue.put(('clear_dataset', None))
        self.got_task.set()
        return self.result_queue.get()

    def close(self):
        self.queue.put(('exit', None))
        self.got_task.set()
        self.loading_process.join()

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

    if num_turns == 0:
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

        if turn == "":
            empty = True
            return conversation_tokenized, conversation_content_ranges, empty

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

def tokenize_dataset(dataset_path, model_path, prompt_format, context_len, save_sys_range, save_user_range, save_assistant_range, add_bos=True) -> tuple[list[ConvoTokenized], set[int]]:
    try:
        config = ExLlamaV2Config()
        config.model_dir = model_path
        config.prepare()
        tokenizer = ExLlamaV2Tokenizer(config)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)

    sp_toks = get_special_tokens(model_path=model_path)
    pf = encode_prompt_format(prompt_format, sp_toks, tokenizer)
    
    dataset = []
    ids = set()

    for convo_id, item in tqdm(enumerate(read_jsonl_lazy(dataset_path)), desc="Tokenizing", unit="convo", smoothing=0.06, leave=False): # Every conversation
        conversation_tokenized, conversation_content_ranges, empty = tokenize_convo(item, sp_toks, tokenizer, pf, save_sys_range, save_user_range, save_assistant_range, context_len, add_bos=add_bos)
        
        if empty:
            continue

        num_pad_tokens = 0
        cropped_end = False

        if len(conversation_tokenized) > context_len:
            conversation_tokenized = conversation_tokenized[:context_len]
        else:
            num_pad_tokens = context_len - len(conversation_tokenized)
            conversation_tokenized = np.concatenate((conversation_tokenized, np.array([sp_toks["eos_id"]] * num_pad_tokens, dtype=np.int64)))

        corrected_content_ranges = []
        for start, end in conversation_content_ranges:
            if start > context_len:
                break
            if end > context_len:
                end = context_len
                cropped_end = True
            corrected_content_ranges.append((start, end))

        conversation = ConvoTokenized(conversation_tokenized, corrected_content_ranges, num_pad_tokens, cropped_end, convo_id)
        ids.add(convo_id)
        dataset.append(conversation)

    return dataset, ids