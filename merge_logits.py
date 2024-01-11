import os
from numpy import save
import torch
import pickle
import threading
from utils.finetuning_utils import teacher_tensors_hander
from utils.dataset_utils import load_metadata

def preprocess_logits(distributions_tensor: torch.Tensor, device: str, metadata: dict):
    added_tokens_ids = metadata.get("added_tokens_ids", [])
    if added_tokens_ids:
        mask = ~torch.isin(torch.arange(distributions_tensor.size(0)), torch.tensor(added_tokens_ids, device=device))
        distributions_tensor = distributions_tensor[mask]
    return distributions_tensor

def check_errors(model_folders_paths: list):
    merged_metadata = load_metadata(model_folders_paths[0])

    flags_to_check = ["sorted", "save_sys_range", "save_user_range", "save_assistant_range"]

    for model_folder_path in model_folders_paths:
        model_metadata = load_metadata(model_folder_path)

        if model_metadata["vocab_family"] != merged_metadata["vocab_family"]:
            raise ValueError(f"Vocab family mismatch in {model_folder_path}")

        for flag in flags_to_check:
            if model_metadata[flag] != merged_metadata[flag]:
                raise ValueError(f"{flag} mismatch in {model_folder_path}\nModel metadata: {model_metadata}\nMerged metadata: {merged_metadata}")
        
def async_save_merged_logits(merged_logits_list: list, save_folder: str, count: int):
    merged_logits_cpu = [logit.numpy() for logit in merged_logits_list]
    save_path = os.path.join(save_folder, f"merged_logits_{count}.pkl")

    def save_thread():
        with open(save_path, 'wb') as f:
            pickle.dump(merged_logits_cpu, f)

    thread = threading.Thread(target=save_thread)
    thread.start()

def merge_and_save_logits(model_folders: list, device: str, output_folder: str, max_cache_gb: int):
    TOKEN_DISTRIBUTION_SIZE_KB = 62.5
    MAX_CACHE_SIZE_KB = max_cache_gb * 1048576  # 1GB = 1048576KB
    MAX_CACHE_SIZE_TOKENS = MAX_CACHE_SIZE_KB / TOKEN_DISTRIBUTION_SIZE_KB

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    model_metadatas = [load_metadata(model_folder) for model_folder in model_folders]

    generators = [teacher_tensors_hander(model_folder, device, loop=False) for model_folder in model_folders]

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

model_folders = [os.path.join(distiributions_folders_path, model_name) 
                for model_name in os.listdir(distiributions_folders_path) 
                if os.path.isdir(os.path.join(distiributions_folders_path, model_name))]

if len(model_folders) < 2:
    raise ValueError("At least 2 models are required to merge")

check_errors(model_folders)
merge_and_save_logits(model_folders, device, output_folder, max_cache_gb)