from exllamav2 import ExLlamaV2Tokenizer, ExLlamaV2Config
from utils.vocab_utils import get_special_tokens
from classes.data_classes import ConvoTokenized
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
    
    if not conversation_content_ranges:
        empty = True
        return conversation_tokenized, conversation_content_ranges, empty

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

    if not dataset:
        raise ValueError("No conversations found in dataset.")
    
    return dataset, ids