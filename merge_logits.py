import os
import torch
import pickle
import threading
from utils.finetuning_utils import teacher_tensors_hander

def async_save_merged_logits(merged_logits_list: list, save_folder: str, count: int):
    merged_logits_cpu = [logit.numpy() for logit in merged_logits_list]
    save_path = os.path.join(save_folder, f"merged_logits_{count}.pkl")

    def save_thread():
        with open(save_path, 'wb') as f:
            pickle.dump(merged_logits_cpu, f)

    thread = threading.Thread(target=save_thread)
    thread.start()

def merge_and_save_logits(all_distributions_folder: str, device: str, output_folder: str, max_cache_gb: int):
    TOKEN_DISTRIBUTION_SIZE_KB = 62.5
    MAX_CACHE_SIZE_KB = max_cache_gb * 1048576  # 1GB = 1048576KB
    MAX_CACHE_SIZE_TOKENS = MAX_CACHE_SIZE_KB / TOKEN_DISTRIBUTION_SIZE_KB

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_folders = [os.path.join(all_distributions_folder, model_name) 
                     for model_name in os.listdir(all_distributions_folder) 
                     if os.path.isdir(os.path.join(all_distributions_folder, model_name))]

    model_metadatas = []

    generators = [teacher_tensors_hander(model_folder, device) for model_folder in model_folders]

    merged_logits_list = []
    current_cache_size_tokens = 0
    count = 0

    try:
        while True:
            logits_list = [next(gen) for gen in generators]
            merged_logits = torch.stack(logits_list).mean(dim=0)
            merged_logits_list.append(merged_logits)

            current_cache_size_tokens += merged_logits.size(0)
            if current_cache_size_tokens >= MAX_CACHE_SIZE_TOKENS:
                count += 1
                async_save_merged_logits(merged_logits_list, output_folder, count)
                merged_logits_list = []
                current_cache_size_tokens = 0

    except StopIteration:
        if merged_logits_list:
            count += 1
            async_save_merged_logits(merged_logits_list, output_folder, count)

    print(f"Merged logits saved in {output_folder}")


distiributions_folders_path = r"F:\distilled\randoBS"
output_folder = r"F:\distilled\merged_logits"
device = "cuda:0"
max_cache_gb = 10
merge_and_save_logits(distiributions_folders_path, device, output_folder, max_cache_gb)