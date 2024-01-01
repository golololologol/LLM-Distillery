import re
import torch
import json
from transformers import AutoTokenizer

def tokenize_dataset(dataset_path, device, sort, model_path, prompt_format, context_len, save_sys_range=False, save_user_range=False, save_assistant_range=False):

    def read_jsonl_lazy(file_path): # Generator to lazy read the dataset line-by-line
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                yield json.loads(line)

    print("Tokenizing the dataset...")
    total_tokens = 0

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def good_encode(text: str, encode_special = True, replace_tokens = True) -> torch.Tensor:
        if replace_tokens:
            text = text.replace('<bos>', tokenizer.bos_token).replace('<eos>', tokenizer.eos_token)
        return tokenizer.encode("\n" + text, add_special_tokens=False, return_tensors="pt").squeeze(0)[2:] # type: ignore
    
    dataset_tokenized = []
    dataset_content_ranges = []
    sys_start_tokenized = good_encode(prompt_format['SYS_START'])
    sys_end_tokenized = good_encode(prompt_format['SYS_END'])
    user_start_tokenized = good_encode(prompt_format['USER_START'])
    user_end_tokenized = good_encode(prompt_format['USER_END'])
    assistant_start_tokenized = good_encode(prompt_format['ASSISTANT_START'])
    assistant_end_tokenized = good_encode(prompt_format['ASSISTANT_END'])

    for item in read_jsonl_lazy(dataset_path):  # Every conversation
        conversation_tokenized = torch.Tensor().to(device)
        conversation_content_ranges = []
        start_index = 0

        sys_content = item.get("init", "")
        if sys_content:
            sys_content_tokenized = good_encode(sys_content.strip(), replace_tokens=False)
            sys_tokenized = torch.cat((sys_start_tokenized, sys_content_tokenized, sys_end_tokenized))
            conversation_tokenized = sys_tokenized
            if save_sys_range:
                conversation_content_ranges.append((sys_start_tokenized.numel()-1, sys_tokenized.numel() - sys_end_tokenized.numel()))
            start_index = sys_tokenized.numel()

        for i, turn in enumerate(item["conversations"]):  # Every turn
            assistant = i % 2

            turn_start_tokenized = assistant_start_tokenized if assistant else user_start_tokenized
            turn_end_tokenized = assistant_end_tokenized if assistant else user_end_tokenized
            turn_tokenized = good_encode(turn.strip(), replace_tokens=False)

            full_turn_tokenized = torch.cat((turn_start_tokenized, turn_tokenized, turn_end_tokenized))
            end_index = start_index + full_turn_tokenized.numel()

            if save_assistant_range and assistant:
                content_start_index = start_index + assistant_start_tokenized.numel() - 1
                content_end_index = end_index - assistant_end_tokenized.numel()
                conversation_content_ranges.append((content_start_index, content_end_index))
            elif save_user_range and not assistant:
                content_start_index = start_index + user_start_tokenized.numel() - 1
                content_end_index = end_index - user_end_tokenized.numel()
                conversation_content_ranges.append((content_start_index, content_end_index))

            start_index = end_index

            conversation_tokenized = torch.cat((conversation_tokenized, full_turn_tokenized)) if conversation_tokenized.numel() > 0 else full_turn_tokenized
        total_tokens += conversation_tokenized.numel()
        dataset_tokenized.append(conversation_tokenized[:context_len].to("cpu"))
        dataset_content_ranges.append(conversation_content_ranges)

    if sort:
        combined_list = list(zip(dataset_tokenized, dataset_content_ranges))
        combined_list.sort(key=lambda x: x[0].shape[0], reverse=True)
        dataset_tokenized, dataset_content_ranges = map(list, zip(*combined_list))

    print(f"Total processed tokens: {total_tokens}")

    def get_vocab_family() -> str:
        vocab_family = "Unknown"

        if tokenizer.vocab_size == 30000:
            vocab_family = "GPT2"
        elif tokenizer.vocab_size == 32000:
            token_29999 = tokenizer.convert_ids_to_tokens(29999)
            if token_29999 == "监":
                vocab_family = "Mistral"
            elif token_29999 == "Z":
                vocab_family = "LLAMA"
        elif tokenizer.vocab_size == 50265:
            vocab_family = "GPTNeo"
        elif tokenizer.vocab_size == 50257:
            vocab_family = "GPTJ"
        return vocab_family

    content_decoded = ""
    for range in dataset_content_ranges[0]:
        content_decoded += tokenizer.decode(dataset_tokenized[0][range[0]:range[1]])
        
    vocab = {k: v for k, v in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])}
    metadata = {
            "first_content_decoded": content_decoded,
            "sorted": sort,
            "save_sys_range": save_sys_range,
            "save_user_range": save_user_range,
            "save_assistant_range": save_assistant_range,
            "context_len": context_len,
            "bos_id": tokenizer.bos_token_id,
            "eos_id": tokenizer.eos_token_id,
            "pad_id": tokenizer.pad_token_id,
            "unk_id": tokenizer.unk_token_id,
            "vocab_size": tokenizer.vocab_size,
            "vocab_family": get_vocab_family(),
            "vocab": vocab
            }
    
    return dataset_tokenized, dataset_content_ranges, metadata