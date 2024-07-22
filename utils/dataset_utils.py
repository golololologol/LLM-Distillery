from concurrent.futures import ThreadPoolExecutor, as_completed
from exllamav2 import ExLlamaV2Tokenizer, ExLlamaV2Config
from classes.data_classes import ConvoTokenized
from utils.vocab_utils import get_family_bos_id
from classes.base_model import BaseModel
from transformers import AutoTokenizer
from multiprocessing import cpu_count
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


def good_encode(text: str, tokenizer, encode_special=False):
    if tokenizer.__class__.__name__ != "ExLlamaV2Tokenizer":
        encoded_ids = tokenizer.encode("\n" + text, add_special_tokens=False)[2:]
    else:
        encoded_ids = tokenizer.encode("\n" + text, encode_special_tokens=encode_special).squeeze(0)[2:] # type: ignore

    encoded_text = np.array(encoded_ids, dtype=np.int64)

    return encoded_text


def encode_prompt_format(prompt_format: dict, tokenizer) -> dict[str, list[int]]:
    prompt_format = prompt_format.copy()

    for key, value in prompt_format.items():
        prompt_format[key] = good_encode(value, tokenizer, encode_special=True)
    return prompt_format


def tokenize_sample(args):
    convo_id, json_item, tokenizer, pf, bos_token_id, save_sys_range, save_user_range, save_assistant_range, context_len, model_completion, model_student, model_add_bos, ignore_model_type = args

    conversation_tokenized = np.zeros(context_len, dtype=np.int64)

    if model_add_bos and not bos_token_id is None:
        conversation_tokenized[0] = bos_token_id
        num_toks = 1
        start_idx = 0
    else:
        num_toks = 0
        start_idx = -1

    conversation_content_ranges = []
    tags = json_item.get("tags", [])
    empty = False
    completion = "completion" in tags
    student = model_student or ignore_model_type

    if completion:
        if model_completion or student:
            turn = json_item.get("conversations", [""])[0]
            text = turn.get("value", "")
            if text:
                text_tokenized = good_encode(text.strip(), tokenizer)[:context_len - num_toks]
                conversation_tokenized[num_toks:num_toks + len(text_tokenized)] = text_tokenized
                num_toks += len(text_tokenized)
                conversation_content_ranges.append((0, num_toks))
                return convo_id, conversation_tokenized, conversation_content_ranges, num_toks, empty
            else:
                empty = True

        else:
            empty = True
    
    elif model_completion and not student:
        empty = True

    num_turns = len(json_item["conversations"])

    if num_turns == 0:
        empty = True

    if empty:
        return convo_id, conversation_tokenized, conversation_content_ranges, num_toks, empty
    

    sys_content = json_item.get("init", "")
    if sys_content:
        sys_content_tokenized = good_encode(sys_content.strip(), tokenizer)
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

    for turn in json_item["conversations"]:

        turn_speaker = turn.get("from", "")
        turn_text = turn.get("value", "")

        if not turn_speaker or not turn_text:
            empty = True
            return convo_id, conversation_tokenized, conversation_content_ranges, num_toks, empty

        assistant = turn_speaker == "gpt"

        turn_start_tokenized = pf['ASSISTANT_START'] if assistant else pf['USER_START']
        turn_end_tokenized = pf['ASSISTANT_END'] if assistant else pf['USER_END']
        turn_tokenized = good_encode(turn_text.strip(), tokenizer)
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

    pf = encode_prompt_format(model.prompt_format, tokenizer)
    
    dataset = []
    ids = set()
    bos_token_id = get_family_bos_id(model.vocab_family, tokenizer)
    model_completion = model.completion
    model_student = model.student
    model_add_bos = model.add_bos

    def generate_tasks():
        for convo_id, item in enumerate(read_jsonl(dataset_path)):
            yield (convo_id, item, tokenizer, pf, bos_token_id, save_sys_range, save_user_range, save_assistant_range, context_len, model_completion, model_student, model_add_bos, ignore_model_type)

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