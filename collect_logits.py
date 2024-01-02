import json
import os
import threading
import pickle
import torch
from tqdm import tqdm
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
from utils.dataset_utils import save_dataset_and_metadata, tokenize_dataset

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

def generate_probability_distributions(dataset_tokenized, dataset_content_ranges, model, device, max_cache_gb, next_token_prob_boost):
    TOKEN_DISTRIBUTION_SIZE_KB = 62.5
    MAX_CACHE_SIZE_KB = max_cache_gb * 1048576  # 1GB = 1048576KB
    MAX_CACHE_SIZE_TOKENS = MAX_CACHE_SIZE_KB / TOKEN_DISTRIBUTION_SIZE_KB

    def modify_distributions(model_output, conversation_tokenized) -> torch.Tensor:
        modified_output = model_output.clone()
        for i in range(conversation_tokenized.size(0) - 1):
            next_token = conversation_tokenized[i + 1]
            current_distribution = modified_output[i]
            if set_max_token_prob:
                max_logit = torch.max(current_distribution)
                current_distribution[next_token] = max_logit * next_token_prob_boost
            else:
                current_distribution[next_token] *= next_token_prob_boost
        return modified_output

    def process_conversation(conversation_tokenized: torch.Tensor, conversation_content_ranges: list):
        conv_tokenized_gpu = conversation_tokenized.to(device)[:window_size]
        conv_distributions = model.forward(conv_tokenized_gpu.unsqueeze(0)).squeeze(0)
        if not next_token_prob_boost <= 1:
            conv_distributions = modify_distributions(conv_distributions, conv_tokenized_gpu)
        modified_distributions_cpu = conv_distributions.to("cpu")
        return [modified_distributions_cpu[start:end] for start, end in conversation_content_ranges]

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
model_path = r"C:\Users\gololo\Desktop\speechless-llama2-hermes-orca-platypus-wizardlm-13b-exl2"
dataset_path = r"f:\down\vsakoye\randoBS.jsonl"
distributions_save_folder = r"F:\distilled"
max_input_len = 8192
window_size = 7*1024
max_cache_gb = 10  # Desired size of the saved files in GB

set_max_token_prob = True
next_token_prob_boost = 1.05  # Boost the probability of the next token by this factor
sort = True # Sort by length, stops Vram spikes for some reason. Top - longest, Bottom - shortest

save_sys_range = False
save_user_range = False
save_assistant_range = True

prompt_format = {
    'SYS_START': "### System:\n",
    'USER_START': "",
    'ASSISTANT_START': "### Assistant:\n",
    'SYS_END': '\n\n',
    'USER_END': '\n\n',
    'ASSISTANT_END': '<eos>\n\n' # Use <eos> and <bos> for model-specific special tokens
}

device = "cuda:0"
model_name = os.path.basename(os.path.normpath(model_path))
dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
save_folder = os.path.join(distributions_save_folder, dataset_name, model_name)

print(f"Model: {model_name}\nDataset: {dataset_name}")

model = load_model(model_path, max_input_len)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

additional_data = {
    'sorted': sort,
    'save_sys_range': save_sys_range,
    'save_user_range': save_user_range,
    'save_assistant_range': save_assistant_range,
    'context_len': window_size,
    'next_token_prob_boost': next_token_prob_boost,
    'set_max_token_prob': set_max_token_prob
}

dataset_tokenized, dataset_content_ranges, metadata = tokenize_dataset(dataset_path, device, sort, model_path, prompt_format, save_sys_range, save_user_range, save_assistant_range)
#add additional data to metadata
metadata = {**additional_data, **metadata}
save_dataset_and_metadata(dataset_tokenized, dataset_content_ranges, metadata, save_folder)
generate_probability_distributions(dataset_tokenized, dataset_content_ranges, model, device, max_cache_gb, next_token_prob_boost)

print("Done!\nIf the script didn't close yet, that means its still writing to disk!\nDO NOT STOP IT MANUALLY!")