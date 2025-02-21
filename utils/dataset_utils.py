from exllamav2 import ExLlamaV2Tokenizer, ExLlamaV2Config
from classes.data_classes import ConvoTokenized
from utils.vocab_utils import get_family_bos_id
from classes.base_model import BaseModel
from transformers import AutoTokenizer
from multiprocessing import cpu_count
from joblib import Parallel, delayed
from tqdm import tqdm
import numpy as np
import logging
import json
import gc

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


def read_jsonl(file_path) -> list[dict]:
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


def good_encode(text: str, tokenizer: ExLlamaV2Tokenizer|AutoTokenizer, encode_special=False):
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
    # Break out arguments for clarity
    convo_id, json_item, tokenizer, pf, bos_token_id, save_sys_range, save_user_range, \
        save_assistant_range, context_len, model_completion, model_student, \
        model_add_bos, ignore_model_type, all_not_add_bos = args

    # Prepare an array to hold the tokenized conversation
    conversation_tokenized = np.zeros(context_len, dtype=np.int64)

    if model_add_bos and bos_token_id is not None:
        # If configured, place a BOS token at the start
        # and adjust indexing for subsequent tokens
        conversation_tokenized[0] = bos_token_id
        num_toks = 1
        start_idx = 0
        end_crop = 0
    else:
        # Without BOS, shift the start index and num tokens by 1
        # and possibly reserve end space to make sure overlength convos
        # have the same ammount of content tokens regardless of BOS pesence
        num_toks = 0
        start_idx = -1
        end_crop = 0 if all_not_add_bos else 1

    # Track where specific conversation segments start and end
    conversation_content_ranges = []
    tags = json_item.get("tags", [])
    empty = False
    completion = "completion" in tags
    student = model_student or ignore_model_type

    if completion:
        # If the conversation is flagged for "completion" usage,
        # we handle the first conversation turn directly
        if model_completion or student:
            turn = json_item.get("conversations", [""])[0]
            text = turn.get("value", "")
            if text:
                text_tokenized = good_encode(text.strip(), tokenizer)[:context_len - num_toks - end_crop]
                conversation_tokenized[num_toks:num_toks + len(text_tokenized)] = text_tokenized
                conversation_content_ranges.append((num_toks, num_toks + len(text_tokenized)))
                num_toks += len(text_tokenized)
                return convo_id, conversation_tokenized, conversation_content_ranges, num_toks, empty
            else:
                empty = True

        else:
            empty = True
    
    elif model_completion and not student:
        # If the model is strictly for completion,
        # and we're not ignoring model type, skip if no "completion" tag
        empty = True

    # If there's no conversation content or we're flagged empty, return
    num_turns = len(json_item["conversations"])

    if num_turns == 0:
        empty = True

    if empty:
        return convo_id, conversation_tokenized, conversation_content_ranges, num_toks, empty
    

    sys_content = json_item.get("init", "")
    if sys_content:
        # Insert system prompt text, wrapping it with SYS_START / SYS_END
        sys_content_tokenized = good_encode(sys_content.strip(), tokenizer)
        sys_tokenized = np.zeros(len(pf['SYS_START']) + len(sys_content_tokenized) + len(pf['SYS_END']), dtype=np.int64)
        sys_tokenized[:len(pf['SYS_START'])] = pf['SYS_START']
        sys_tokenized[len(pf['SYS_START']):len(pf['SYS_START']) + len(sys_content_tokenized)] = sys_content_tokenized
        sys_tokenized[len(pf['SYS_START']) + len(sys_content_tokenized):] = pf['SYS_END']
        conversation_tokenized[num_toks:num_toks + len(sys_tokenized)] = sys_tokenized[:context_len - num_toks]
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

        # For each turn in the conversation:
        # Determine if it's user or assistant,
        # tokenize, and wrap with start/end tokens
        turn_start_tokenized = pf['ASSISTANT_START'] if assistant else pf['USER_START']
        turn_end_tokenized = pf['ASSISTANT_END'] if assistant else pf['USER_END']
        turn_tokenized = good_encode(turn_text.strip(), tokenizer)
        turn_len = len(turn_tokenized)

        full_turn_tokenized = np.zeros(len(turn_start_tokenized) + turn_len + len(turn_end_tokenized), dtype=np.int64)
        full_turn_tokenized[:len(turn_start_tokenized)] = turn_start_tokenized
        full_turn_tokenized[len(turn_start_tokenized):len(turn_start_tokenized) + turn_len] = turn_tokenized
        full_turn_tokenized[len(turn_start_tokenized) + turn_len:] = turn_end_tokenized
        end_idx = start_idx + len(full_turn_tokenized)

        if (save_assistant_range and assistant) or (save_user_range and not assistant):
            # save relevant content ranges for later use
            content_start_index = start_idx + len(turn_start_tokenized)
            content_end_index = content_start_index + turn_len + 1
            conversation_content_ranges.append((content_start_index, content_end_index))

        start_idx = end_idx

        if num_toks + len(full_turn_tokenized) > context_len:
            # Stop if adding tokens would exceed the maximum allowed length
            # and break out of the loop
            conversation_tokenized[num_toks:context_len - end_crop] = full_turn_tokenized[:context_len - num_toks - end_crop]
            num_toks = context_len - end_crop
            break

        # Otherwise, place the turn tokens into the main conversation array
        conversation_tokenized[num_toks:num_toks + len(full_turn_tokenized)] = full_turn_tokenized
        num_toks += len(full_turn_tokenized)
    
    if not conversation_content_ranges:
        # If no ranges were recorded, the conversation ends up empty
        empty = True
        return convo_id, conversation_tokenized, conversation_content_ranges, num_toks, empty

    if conversation_content_ranges[0][0] > context_len:
        # If the first content range is beyond what fits, it's all empty
        empty = True

    return convo_id, conversation_tokenized, conversation_content_ranges, num_toks, empty

def tokenize_dataset(dataset_path, context_len, save_sys_range, save_user_range, save_assistant_range, model: BaseModel, ignore_model_type, all_not_add_bos) -> tuple[list[ConvoTokenized], set[int]]:
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
    id_to_sample = {convo_id: item for convo_id, item in enumerate(read_jsonl(dataset_path))}
    num_samples = len(id_to_sample)
    num_cores = cpu_count()
    prefer = 'processes' if num_samples > 2048 else 'threads'
    
    results = Parallel(n_jobs=num_cores, batch_size=num_samples//num_cores, pre_dispatch=num_samples//10, prefer=prefer, return_as="generator")(
        delayed(tokenize_sample)(
            (
                convo_id, item, tokenizer, pf, bos_token_id, 
                save_sys_range, save_user_range, save_assistant_range, 
                context_len, model_completion, model_student, 
                model_add_bos, ignore_model_type, all_not_add_bos
            )
        ) for convo_id, item in id_to_sample.items()
    )
    
    for convo_id, conversation_tokenized, conversation_content_ranges, num_toks, empty in tqdm(results, total=num_samples, desc="Tokenizing", unit="convo", leave=False):
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
        
        conversation = ConvoTokenized(
            conversation_tokenized, 
            corrected_content_ranges, 
            num_pad_tokens, 
            cropped_end, 
            convo_id, 
            str(id_to_sample[convo_id])
        )
        ids.add(convo_id)
        dataset.append(conversation)

    if not dataset:
        raise ValueError("No conversations were tokenized.\nIf you are using a completion model, make sure the completion tag is present in the conversations of your dataset.")
    
    dataset.sort(key=lambda x: x.origin_convo_id)

    return dataset, ids