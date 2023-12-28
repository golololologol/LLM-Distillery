import json
import os
import threading
import pickle
from numpy import save
import torch
from tqdm import tqdm
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
from tokenize_dataset import tokenize_dataset

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
    cache = ExLlamaV2Cache(model, lazy=True)

    model.load_autosplit(cache)
    return model

def save_tokenized_dataset(dataset_tokenized: list, dataset_content_ranges: list, metadata, save_metadata: bool):
    file = os.path.join(save_folder, "dataset_tokenized.jsonl")
    metadata_file = os.path.join(save_folder, "dataset_metadata.json")

    with open(file, 'w', encoding='utf-8') as f, open(metadata_file, 'w', encoding='utf-8') as meta_f:
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

        if save_metadata:
            meta_f.write(json.dumps(metadata, ensure_ascii=False) + '\n')

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
save_metadata = True
save_sys_range = True
save_user_range = True
save_assistant_range = True


prompt_format = {
    'SYS_START': "### System:\n",
    'USER_START': "### User:\n",
    'ASSISTANT_START': "### Assistant:\n",
    'SYS_END': '\n',
    'USER_END': '\n',
    'ASSISTANT_END': '<eos>\n' # Use <eos> and <bos> for model-specific special tokens
}

device = "cuda:0"
model_name = os.path.basename(os.path.normpath(model_path))
dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
save_folder = os.path.join(distributions_save_folder, dataset_name, model_name)

print(f"Model: {model_name}\nDataset: {dataset_name}")

model = load_model(model_path, max_input_len)
dataset = read_jsonl(dataset_path)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

dataset_tokenized, dataset_content_ranges, metadata = tokenize_dataset(dataset, device, sort, model_path, save_metadata, prompt_format, save_sys_range, save_user_range, save_assistant_range)
save_tokenized_dataset(dataset_tokenized, dataset_content_ranges, metadata, save_metadata)
generate_probability_distributions(dataset_tokenized, dataset_content_ranges, device)

print("Done!\nIf the script didn't close yet, that means its still writing to disk!\nDO NOT STOP IT MANUALLY!")