from concurrent.futures import ThreadPoolExecutor, as_completed
from exllamav2 import ExLlamaV2Tokenizer, ExLlamaV2Config
from utils.vocab_utils import get_special_tokens
from classes.data_classes import ConvoTokenized
from multiprocessing import cpu_count
from classes.base_model import BaseModel
from transformers import AutoTokenizer
from tqdm import tqdm
import numpy as np
import logging
import json

logging.getLogger("transformers").setLevel(logging.ERROR)  # Shut up transformers
logging.getLogger("torch").setLevel(logging.ERROR) # And pytorch for good measure


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


def read_jsonl(file_path):
    data_list = []
    with open(file_path, 'r', encoding='utf-8') as f:
        line_number = 0
        for line in f:
            line_number += 1
            try:
                data = json.loads(line)
                data_list.append(data)
            except json.JSONDecodeError as e:
                print(f"Fuck up on line {line_number}: {e.msg}. Line content: {line.strip()}")
    return data_list


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

def tokenize_sample(args):
    convo_id, json_item, sp_toks, tokenizer, pf, save_sys_range, save_user_range, save_assistant_range, context_len, model, ignore_model_type = args

    conversation_tokenized = np.zeros(context_len, dtype=np.int64)

    if model.add_bos:
        conversation_tokenized[0] = sp_toks["bos_id"]
        num_toks = 1
        start_idx = 0
    else:
        num_toks = 0
        start_idx = -1

    conversation_content_ranges = []
    tags = json_item.get("tags", [])
    empty = False
    completion = "completion" in tags
    student = model.student or ignore_model_type

    if completion:
        if model.completion or student:
            text = json_item.get("conversations", [""])[0]

            if text:
                text_tokenized = good_encode(text.strip(), sp_toks, tokenizer, replace_tokens=False, encode_special=False)[:context_len - num_toks]
                conversation_tokenized[num_toks:num_toks + len(text_tokenized)] = text_tokenized
                num_toks += len(text_tokenized)
                conversation_content_ranges.append((0, num_toks))
                return convo_id, conversation_tokenized, conversation_content_ranges, num_toks, empty
            else:
                empty = True

        else:
            empty = True
    
    elif model.completion and not student:
        empty = True

    num_turns = len(json_item["conversations"])

    if num_turns == 0:
        empty = True

    if empty:
        return convo_id, conversation_tokenized, conversation_content_ranges, num_toks, empty
    
    
    reversed = ("reversed" in tags) or json_item.get("reversed", False)

    sys_content = json_item.get("init", "")
    if sys_content:
        sys_content_tokenized = good_encode(sys_content.strip(), sp_toks, tokenizer, replace_tokens=False, encode_special=False)
        sys_tokenized = np.zeros(len(pf['SYS_START']) + len(sys_content_tokenized) + len(pf['SYS_END']), dtype=np.int64)
        sys_tokenized[:len(pf['SYS_START'])] = pf['SYS_START']
        sys_tokenized[len(pf['SYS_START']):len(pf['SYS_START']) + len(sys_content_tokenized)] = sys_content_tokenized
        sys_tokenized[len(pf['SYS_START']) + len(sys_content_tokenized):] = pf['SYS_END']
        sys_tokenized = sys_tokenized[:context_len - num_toks]
        conversation_tokenized[num_toks:num_toks + len(sys_tokenized)] = sys_tokenized
        num_toks += len(sys_tokenized)
        if save_sys_range:
            conversation_content_ranges.append((len(pf['SYS_START']) + start_idx, num_toks - len(pf['SYS_END'])))
        start_idx = num_toks - 1

    for i, turn in enumerate(json_item["conversations"]):

        if turn == "":
            empty = True
            return convo_id, conversation_tokenized, conversation_content_ranges, num_toks, empty

        if reversed:
            assistant = (i + 1) % 2
        else:
            assistant = i % 2

        turn_start_tokenized = pf['ASSISTANT_START'] if assistant else pf['USER_START']
        turn_end_tokenized = pf['ASSISTANT_END'] if assistant else pf['USER_END']
        turn_tokenized = good_encode(turn.strip(), sp_toks, tokenizer, replace_tokens=False, encode_special=False)
        turn_len = len(turn_tokenized)

        full_turn_tokenized = np.zeros(len(turn_start_tokenized) + turn_len + len(turn_end_tokenized), dtype=np.int64)
        full_turn_tokenized[:len(turn_start_tokenized)] = turn_start_tokenized
        full_turn_tokenized[len(turn_start_tokenized):len(turn_start_tokenized) + turn_len] = turn_tokenized
        full_turn_tokenized[len(turn_start_tokenized) + turn_len:] = turn_end_tokenized
        end_idx = start_idx + len(full_turn_tokenized)

        if save_assistant_range and assistant:
            content_start_index = start_idx + len(pf['ASSISTANT_START'])
            content_end_index = content_start_index + turn_len + 1
            conversation_content_ranges.append((content_start_index, content_end_index))
        if save_user_range and not assistant:
            content_start_index = start_idx + len(pf['USER_START'])
            content_end_index = content_start_index + turn_len + 1
            conversation_content_ranges.append((content_start_index, content_end_index))

        start_idx = end_idx

        if num_toks + len(full_turn_tokenized) > context_len:
            conversation_tokenized[num_toks:] = full_turn_tokenized[:context_len - num_toks]
            num_toks = context_len
            break

        conversation_tokenized[num_toks:num_toks + len(full_turn_tokenized)] = full_turn_tokenized
        num_toks += len(full_turn_tokenized)
    
    if not conversation_content_ranges:
        empty = True
        return convo_id, conversation_tokenized, conversation_content_ranges, num_toks, empty

    if conversation_content_ranges[0][0] > context_len:
        empty = True

    return convo_id, conversation_tokenized, conversation_content_ranges, num_toks, empty

def tokenize_dataset(dataset_path, context_len, save_sys_range, save_user_range, save_assistant_range, model: BaseModel, ignore_model_type) -> tuple[list[ConvoTokenized], set[int]]:
    try:
        config = ExLlamaV2Config()
        config.model_dir = model.model_path
        config.prepare()
        tokenizer = ExLlamaV2Tokenizer(config)
    except:
        tokenizer = AutoTokenizer.from_pretrained(model.model_path, legacy=False)

    sp_toks = get_special_tokens(model_path=model.model_path)
    pf = encode_prompt_format(model.prompt_format, sp_toks, tokenizer)
    
    dataset = []
    ids = set()

    def generate_tasks():
        for convo_id, item in enumerate(read_jsonl(dataset_path)):
            yield (convo_id, item, sp_toks, tokenizer, pf, save_sys_range, save_user_range, save_assistant_range, context_len, model, ignore_model_type)

    with ThreadPoolExecutor(max_workers=cpu_count()) as executor:
        futures = {executor.submit(tokenize_sample, task): task for task in generate_tasks()}
        for future in tqdm(as_completed(futures), total=len(futures), desc="Tokenizing", unit="convo", smoothing=0.06, leave=False):
            convo_id, conversation_tokenized, conversation_content_ranges, num_toks, empty = future.result()
            
            if empty:
                continue

            num_pad_tokens = context_len - num_toks
            cropped_end = False

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

    if not dataset:
        raise ValueError("No conversations were tokenized.\nIf you are using a completion model, make sure the completion tag is present in the conversations of your dataset.")
    
    dataset.sort(key=lambda x: x.origin_convo_id)

    return dataset, ids