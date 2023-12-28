import torch
from transformers import AutoTokenizer

def tokenize_dataset(dataset, device, sort, model_path, return_metadata, prompt_format, save_sys_range=False, save_user_range=False, save_assistant_range=False):

    print("Tokenizing the dataset")
    total_tokens = 0

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    def good_encode(text: str, encode_special = True, replace_tokens = False):
        if replace_tokens:
            text = text.replace('<bos>', tokenizer.bos_token).replace('<eos>', tokenizer.eos_token)
        return tokenizer.encode("\n" + text, add_special_tokens=False, return_tensors="pt").squeeze(0)[2:] # type: ignore
    
    dataset_tokenized = []
    dataset_content_ranges = []
    sys_start_tokenized = good_encode(prompt_format['SYS_START'], replace_tokens=True)
    sys_end_tokenized = good_encode(prompt_format['SYS_END'], replace_tokens=True)
    user_start_tokenized = good_encode(prompt_format['USER_START'], replace_tokens=True)
    user_end_tokenized = good_encode(prompt_format['USER_END'], replace_tokens=True)
    assistant_start_tokenized = good_encode(prompt_format['ASSISTANT_START'], replace_tokens=True)
    assistant_end_tokenized = good_encode(prompt_format['ASSISTANT_END'], replace_tokens=True)

    for item in dataset:  # Every conversation
        conversation_tokenized = torch.Tensor().to(device)
        conversation_content_ranges = []
        start_index = 0

        sys_content = item["init"]
        if sys_content:
            
            sys_content_tokenized = good_encode(sys_content.strip(), encode_special=False)
            sys_tokenized = torch.cat((sys_start_tokenized, sys_content_tokenized, sys_end_tokenized))
            conversation_tokenized = sys_tokenized
            if save_sys_range:
                conversation_content_ranges.append((sys_start_tokenized.numel()-1, sys_tokenized.numel() - sys_end_tokenized.numel()))
            start_index = sys_tokenized.numel()

        for i, turn in enumerate(item["conversations"]):  # Every turn
            assistant = i % 2

            turn_start_tokenized = assistant_start_tokenized if assistant else user_start_tokenized
            turn_end_tokenized = assistant_end_tokenized if assistant else user_end_tokenized
            turn_tokenized = good_encode(turn.strip(), encode_special=False)

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

            conversation_tokenized = torch.cat(
                (conversation_tokenized, full_turn_tokenized)) if conversation_tokenized.numel() > 0 else full_turn_tokenized
        total_tokens += conversation_tokenized.numel()
        dataset_tokenized.append(conversation_tokenized.to("cpu"))
        dataset_content_ranges.append(conversation_content_ranges)

    if sort:
        combined_list = list(zip(dataset_tokenized, dataset_content_ranges))
        combined_list.sort(key=lambda x: x[0].shape[0], reverse=True)
        dataset_tokenized, dataset_content_ranges = map(list, zip(*combined_list))

    print(f"Total processed tokens: {total_tokens}")

    metadata_to_save = None
    if return_metadata:
        content_decoded = ""
        for range in dataset_content_ranges[0]:
            content_decoded += tokenizer.decode(dataset_tokenized[0][range[0]:range[1]])
        metadata_to_save = {
                "first_content_decoded": content_decoded,
                "sorted": sort,
                "save_sys_range": save_sys_range,
                "save_user_range": save_user_range,
                "save_assistant_range": save_assistant_range,
                "bos_id": tokenizer.bos_token_id,
                "eos_id": tokenizer.eos_token_id,
                "pad_id": tokenizer.pad_token_id,
                "unk_id": tokenizer.unk_token_id,
                "vocab_size": tokenizer.vocab_size,
                "vocab": {k: v for k, v in sorted(tokenizer.get_vocab().items(), key=lambda item: item[1])}
            }
    
    return dataset_tokenized, dataset_content_ranges, metadata_to_save