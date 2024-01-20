import os
import json
import torch
from transformers import AutoTokenizer

def read_jsonl_lazy(file_path): # Generator to lazy read the dataset line-by-line
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)

def good_encode(text: str, encode_special = True, replace_tokens = True, tokenizer=None, model_path="") -> torch.Tensor:
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    if replace_tokens:
        text = text.replace('<bos>', tokenizer.bos_token).replace('<eos>', tokenizer.eos_token)
    return tokenizer.encode("\n" + text, add_special_tokens=False, return_tensors="pt").squeeze(0)[2:] # type: ignore

def encode_prompt_format(prompt_format: dict, tokenizer=None, model_path="") -> dict:
    if tokenizer == None:
        tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    for key, value in prompt_format.items():
        prompt_format[key] = good_encode(value, tokenizer=tokenizer)
    return prompt_format

def tokenize_dataset(dataset_path, device, sort, model_path, prompt_format, context_len, save_sys_range=False, save_user_range=False, save_assistant_range=False):
    print("Tokenizing the dataset...")
    total_tokens = 0
    empty_convo_ids = []

    tokenizer = AutoTokenizer.from_pretrained(model_path, legacy=False)
    
    pf = encode_prompt_format(prompt_format, tokenizer)

    dataset_tokenized = []
    dataset_content_ranges = []
    num_convos = 0

    for convo_id, item in enumerate(read_jsonl_lazy(dataset_path)):  # Every conversation
        conversation_tokenized = torch.Tensor().to(device)
        num_convos += 1
        conversation_content_ranges = []
        start_index = 0
        
        tags = item.get("tags", [])
        reversed = ("reversed" in tags) or item.get("reversed", False)

        sys_content = item.get("init", "")
        if sys_content:
            sys_content_tokenized = good_encode(sys_content.strip(), replace_tokens=False, encode_special=False, tokenizer=tokenizer)
            sys_tokenized = torch.cat((pf['SYS_START'], sys_content_tokenized, pf['SYS_END']))
            conversation_tokenized = sys_tokenized
            if save_sys_range:
                conversation_content_ranges.append((pf['SYS_START'].numel()-1, sys_tokenized.numel() - pf['SYS_END'].numel()))
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

def generate_metadata(model_path: str, dataset_tokenized: list, dataset_content_ranges: list) -> dict:
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
            "empty_convo_ids": [],
            "merged": False,
            "merged_models": [],
            "model_name": model_path.split("/")[-1],
            "bos_id": tokenizer.bos_token_id,
            "eos_id": tokenizer.eos_token_id,
            "pad_id": tokenizer.pad_token_id,
            "unk_id": tokenizer.unk_token_id,
            "added_tokens_ids": added_tokens_ids,
            "vocab_size": tokenizer.vocab_size,
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
    return ([item for i, item in enumerate(dataset_tokenized) if i not in empty_convo_ids],
            [item for i, item in enumerate(dataset_content_ranges) if i not in empty_convo_ids])

def save_metadata(metadata: dict, save_folder: str):
    metadata_file = os.path.join(save_folder, "dataset_metadata.json")
    with open(metadata_file, 'w', encoding='utf-8') as meta_f:
        json.dump(metadata, meta_f, ensure_ascii=False, indent=4)

def load_metadata(distributions_path: str) -> dict:
    metadata_path = os.path.join(distributions_path, "dataset_metadata.json")
    with open(metadata_path, 'r', encoding='utf-8') as file:
        metadata = json.load(file)
        return metadata
    