import os
import torch
import pickle
import threading
from utils.finetuning_utils import teacher_tensors_hander
from utils.dataset_utils import load_metadata, save_metadata

def check_errors(model_folders_paths: list):
    merged_metadata = load_metadata(model_folders_paths[0])

    flags_to_check = ["sorted", "save_sys_range", "save_user_range", "save_assistant_range"]

    for model_folder_path in model_folders_paths:
        model_metadata = load_metadata(model_folder_path)

        if model_metadata["vocab_family"] != merged_metadata["vocab_family"]:
            raise ValueError(f"Vocab family mismatch in {model_folder_path}\nModel family: {model_metadata['vocab_family']}\nMerged family: {merged_metadata['vocab_family']}")

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

    generators = [teacher_tensors_hander(model_folder, device, loop=False) for model_folder in model_folders]

    model_metadatas = [load_metadata(model_folder) for model_folder in model_folders]

    merged_metadata = model_metadatas[0]
    merged_metadata["vocab_size"] = merged_metadata["vocab_size"] - len(merged_metadata["added_tokens_ids"])
    merged_metadata["merged"] = True
    merged_metadata["context_len"] = max([metadata["context_len"] for metadata in model_metadatas])
    merged_metadata["merged_models"] = [metadata["model_name"] for metadata in model_metadatas]

    merged_probs_list = []
    current_cache_size_tokens = 0
    count = 0
    convo_id = 0

    try:
        while True:
            logprobs_list = [torch.log(next(gen)) for gen in generators]
            logprobs_list = [tensor for tensor in logprobs_list if tensor.shape[0] > 0]
            empty = True if not logprobs_list else False

            if not empty:
                logprobs_list.sort(key=lambda x: x.shape[0], reverse=True)
                merged_logprobs = logprobs_list[0]

                for tensor in logprobs_list[1:]:
                    merge_length = min(merged_logprobs.shape[0], tensor.shape[0])

                    merged_part = torch.stack([merged_logprobs[:merge_length], tensor[:merge_length]]).mean(dim=0)

                    if merged_logprobs.shape[0] > merge_length:
                        unmerged_part = merged_logprobs[merge_length:]
                        merged_logprobs = torch.cat([merged_part, unmerged_part], dim=0)
                    else:
                        merged_logprobs = merged_part
                merged_probs = torch.exp(merged_logprobs)
                merged_probs /= merged_probs.sum()
            else:
                merged_probs = torch.tensor([], device=device)
                merged_metadata["empty_convo_ids"].append(convo_id)

            convo_id += 1
            
            merged_probs_list.append(merged_probs)
            current_cache_size_tokens += merged_probs.size(0)

            if current_cache_size_tokens >= MAX_CACHE_SIZE_TOKENS:
                count += 1
                async_save_merged_logits(merged_probs_list, output_folder, count)
                merged_probs_list = []
                current_cache_size_tokens = 0

    except StopIteration:
        save_metadata(merged_metadata, output_folder)
        if merged_probs_list:
            count += 1
            async_save_merged_logits(merged_probs_list, output_folder, count)

    print(f"Merged logits saved in {output_folder}")


# Parameters
distributions_folders_path = r"F:\distilled\randoBS"
device = "cuda:0"
max_cache_gb = 10


model_folders = [os.path.join(distributions_folders_path, model_name) 
                for model_name in os.listdir(distributions_folders_path) 
                if os.path.isdir(os.path.join(distributions_folders_path, model_name))]

if len(model_folders) < 2:
    raise ValueError("At least 2 models are required to merge")

output_folder = os.path.join(distributions_folders_path, "merged")

check_errors(model_folders)
merge_and_save_logits(model_folders, device, output_folder, max_cache_gb)