import json
import os
import threading
import pickle
import torch
from tqdm import tqdm
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_model(model_path: str, max_input_len: int):
    print("Loading model...")
    config = ExLlamaV2Config()
    config.model_dir = model_path
    config.prepare()
    config.max_seq_len = max_input_len
    config.max_input_len = max_input_len
    config.max_attention_size = max_input_len**2

    model = ExLlamaV2(config)
    tokenizer = ExLlamaV2Tokenizer(config)
    cache = ExLlamaV2Cache(model, lazy=True)

    model.load_autosplit(cache)
    return model, tokenizer

def save_tokenized_dataset(dataset_tokenized: list, dataset_content_ranges: list):
    file = save_folder + "/dataset_tokenized.jsonl"
    with open(file, 'w', encoding='utf-8') as f:
        for i, convo_tokenized in enumerate(dataset_tokenized):
            
            data_to_save = {
                "convo_tokenized": convo_tokenized.tolist(),
                "content_ranges": dataset_content_ranges[i]
            }

            content_tokens = []
            content_decoded = []
            
            for content_range in dataset_content_ranges[i]:
                content_start, content_end = content_range
                content_tokens.append(convo_tokenized[content_start:content_end].tolist())
                content_decoded.append(tokenizer.decode(convo_tokenized[content_start:content_end], decode_special_tokens=True))

            decoded = {"Decoded": tokenizer.decode(convo_tokenized, decode_special_tokens=True)}
            content_tokens = {"Content_tokens": content_tokens}
            content_decoded = {"Content_decoded": content_decoded}

            f.write(json.dumps(data_to_save, ensure_ascii=False) + '\n')
            f.write(json.dumps(decoded, ensure_ascii=False) + '\n')
            f.write(json.dumps(content_tokens, ensure_ascii=False) + '\n')
            f.write(json.dumps(content_decoded, ensure_ascii=False) + '\n')

def async_save_partial_distributions(dataset_distributions: list, count: int):
    save_path = os.path.join(save_folder, "distributions")
    dataset_distributions_cpu = []
    for convo_distr in dataset_distributions:
        dataset_distributions_cpu.append(convo_distr.numpy())
        
    def save_thread():
        with open(f'{save_path}_{count}.pkl', 'wb') as f:
            pickle.dump(dataset_distributions_cpu, f)

    thread = threading.Thread(target=save_thread)
    thread.start()


def tokenize_dataset(dataset: list):

    def good_encode(text: str, encode_special_tokens = False):
        return tokenizer.encode("\n" + text, encode_special_tokens=encode_special_tokens).squeeze(0)[2:] # type: ignore
    
    dataset_tokenized = []
    dataset_content_ranges = []
    user_start_tokenized = good_encode(USER_START.replace("<bos>", tokenizer.bos_token), encode_special_tokens=True)
    user_end_tokenized = good_encode(USER_END.replace("<eos>", tokenizer.eos_token), encode_special_tokens=True)
    assistant_start_tokenized = good_encode(ASSISTANT_START.replace("<bos>", tokenizer.bos_token), encode_special_tokens=True)
    assistant_end_tokenized = good_encode(ASSISTANT_END.replace("<eos>", tokenizer.eos_token), encode_special_tokens=True)
    content_start_offset = assistant_start_tokenized.numel() - 1
    content_end_offset = assistant_end_tokenized.numel()

    for item in dataset: # Every conversation
        conversation_tokenized = torch.Tensor().to(device)
        conversation_content_ranges = []
        start_index = 0

        sys_content = item["init"]
        if sys_content:
            sys_finalized = SYS_START.replace("<bos>", tokenizer.bos_token) + sys_content.strip() + SYS_END.replace("<eos>", tokenizer.eos_token)
            sys_tokenized = good_encode(sys_finalized, encode_special_tokens=True)
            conversation_tokenized = sys_tokenized
            start_index = sys_tokenized.numel()

        for i, turn in enumerate(item["conversations"]): # Every turn
            assistant = i % 2

            turn_start_tokenized = assistant_start_tokenized if assistant else user_start_tokenized
            turn_end_tokenized = assistant_end_tokenized if assistant else user_end_tokenized
            turn_tokenized = good_encode(turn.strip(), encode_special_tokens=True)

            full_turn_tokenized = torch.cat((turn_start_tokenized, turn_tokenized, turn_end_tokenized))
            end_index = start_index + full_turn_tokenized.numel()

            if assistant:
                content_start_index = start_index + content_start_offset
                content_end_index = end_index - content_end_offset
                conversation_content_ranges.append((content_start_index, content_end_index))

            start_index = end_index

            conversation_tokenized = torch.cat((conversation_tokenized, full_turn_tokenized)) if conversation_tokenized.numel() > 0 else full_turn_tokenized

        dataset_tokenized.append(conversation_tokenized.to("cpu"))
        dataset_content_ranges.append(conversation_content_ranges)

    if sort:
        combined_list = list(zip(dataset_tokenized, dataset_content_ranges))
        combined_list.sort(key=lambda x: x[0].shape[0], reverse=True)
        dataset_tokenized, dataset_content_ranges = map(list, zip(*combined_list))

    return dataset_tokenized, dataset_content_ranges


def generate_probability_distributions(dataset_tokenized, dataset_content_ranges, device):
    TOKEN_DISTRIBUTION_SIZE_KB = 62.5
    MAX_CACHE_SIZE_KB = max_cache_gb * 1048576  # 1GB = 1048576KB
    MAX_CACHE_SIZE_TOKENS = MAX_CACHE_SIZE_KB / TOKEN_DISTRIBUTION_SIZE_KB

    def process_conversation(conversation_tokenized, conversation_content_ranges):
        conv_tokenized_gpu = conversation_tokenized[:window_size].to(device)
        conv_distributions = model.forward(conv_tokenized_gpu.unsqueeze(0)).squeeze(0).to("cpu") # type: ignore
        return [conv_distributions[start:end] for start, end in conversation_content_ranges]

    dataset_distributions = []
    current_cache_size_tokens = 0
    count = 0
    total_saved_tokens = 0

    for conversation_tokenized, conversation_content_ranges in tqdm(zip(dataset_tokenized, dataset_content_ranges), desc="Generating Distributions", unit="convo"):
        content_distributions = process_conversation(conversation_tokenized, conversation_content_ranges)
        conversation_content_distributions = torch.cat(content_distributions, dim=0)
        dataset_distributions.append(conversation_content_distributions)

        total_saved_tokens += conversation_content_distributions.size(0)
        current_cache_size_tokens += conversation_content_distributions.size(0)

        if current_cache_size_tokens >= MAX_CACHE_SIZE_TOKENS:
            count += 1
            async_save_partial_distributions(dataset_distributions, count)
            dataset_distributions = []
            current_cache_size_tokens = 0

    print(f"Total saved tokens: {total_saved_tokens}")
    if dataset_distributions:
        count += 1
        async_save_partial_distributions(dataset_distributions, count)


# Main Script
model_path = r"C:\Users\gololo\Desktop\neural-chat-7b-v3-1-exl2"
dataset_path = r"C:\Users\gololo\Documents\janny\janny_Filteredtest.jsonl"
distributions_save_folder = r"F:\distilled"
max_input_len = 8192
window_size = 7168
max_cache_gb = 10  # Desired size of the saved files in GB
sort = True # Sort by length, stops Vram spikes for some reason. Top - longest, Bottom - shortest

SYS_START = "<|im_start|>system\n"
USER_START = "<|im_start|>user\n"
ASSISTANT_START = "<|im_start|>assistant\n"
SYS_END = "<|im_end|>\n"
USER_END = "<|im_end|>\n"
ASSISTANT_END = "<|im_end|>\n" # Use <eos> when the model needs to stop

device = "cuda:0"
model_name = os.path.basename(os.path.normpath(model_path))
dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
save_folder = os.path.join(distributions_save_folder, dataset_name, model_name)

print(f"Model: {model_name}\nDataset: {dataset_name}")

model, tokenizer = load_model(model_path, max_input_len)
dataset = read_jsonl(dataset_path)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

dataset_tokenized, dataset_content_ranges = tokenize_dataset(dataset)
save_tokenized_dataset(dataset_tokenized, dataset_content_ranges)
generate_probability_distributions(dataset_tokenized, dataset_content_ranges, device)

print("Done!\nIf the script didn't close yet, that means its still writing to disk!\nDO NOT STOP IT MANUALLY!")