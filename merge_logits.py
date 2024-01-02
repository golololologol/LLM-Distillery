import torch
from utils.finetuning_utils import teacher_tensors_hander

def merge_teacher_logits(all_distributions_folder: str, device: str):
    for distribition_folder in all_distributions_folder:
        pass







distiributions_folders_path = r"F:\distilled\randoBS"
device = "cuda:0"
merge_teacher_logits(distiributions_folders_path, device)