import os
import torch
from tqdm import tqdm
from utils.dataset_utils import load_metadata, save_metadata, H5Writer, H5Reader

def check_errors(model_folders_paths: list):
    print("Checking for errors...")
    merged_metadata = load_metadata(model_folders_paths[0])

    flags_to_check = ["sorted", "save_sys_range", "save_user_range", "save_assistant_range", "dataset_len", "vocab_family", "crop_to_size"]

    for model_folder_path in model_folders_paths:
        model_metadata = load_metadata(model_folder_path)

        for flag in flags_to_check:
            if model_metadata[flag] != merged_metadata[flag]:
                raise ValueError(f"{flag} mismatch in {model_folder_path}\nModel metadata: {model_metadata}\nMerged metadata: {merged_metadata}")
               
def merge_and_save_probs(model_folders: list, device: str, output_folder: str):
    print("Initializing merging...")
    readers = [H5Reader(folder, device, timeout=90) for folder in model_folders]
    writer = H5Writer(output_folder, timeout=90)

    model_metadatas = [load_metadata(model_folder) for model_folder in model_folders]

    merged_metadata = model_metadatas[0]
    merged_metadata["empty_convo_ids"] = []
    merged_metadata["merged"] = True
    merged_metadata["context_len"] = max([metadata["context_len"] for metadata in model_metadatas])
    merged_metadata["merged_models"] = [metadata["model_name"] for metadata in model_metadatas]
    
    save_metadata(merged_metadata, output_folder)

    for convo_id in tqdm(range(merged_metadata["dataset_len"]), desc="Merging", unit="convo", smoothing=0.05):
        probs_list = [reader.read_next() for reader in readers]
        probs_list = [tensor for tensor in probs_list if tensor.shape[0] > 0]
        empty = True if not probs_list else False

        if not empty:
            probs_list.sort(key=lambda x: x.shape[0], reverse=True)
            merged_probs = probs_list[0]

            for tensor in probs_list[1:]:
                merge_length = tensor.shape[0]
                merged_probs[:merge_length] += tensor

            merged_probs /= merged_probs.sum(dim=-1, keepdim=True)
        else:
            merged_probs = torch.tensor([], device="cpu")
            merged_metadata["empty_convo_ids"].append(convo_id)

        writer.write_data(merged_probs)
    
    writer.close()
    for reader in readers:
        reader.close()
    print(f"Merged probs saved in {output_folder}")


# Parameters
distributions_folders_path = r"F:\distilled\soup"
device = "cuda:0"

model_folders = [os.path.join(distributions_folders_path, model_name) 
                for model_name in os.listdir(distributions_folders_path)
                if os.path.isdir(os.path.join(distributions_folders_path, model_name)) and model_name != "merged"]

if len(model_folders) < 2:
    raise ValueError("At least 2 models are required to merge")

output_folder = os.path.join(distributions_folders_path, "merged")

if not os.path.exists(output_folder):
        os.makedirs(output_folder)

check_errors(model_folders)
merge_and_save_probs(model_folders, device, output_folder)