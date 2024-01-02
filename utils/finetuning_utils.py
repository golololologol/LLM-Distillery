import torch.nn as nn
import torch
import os
import pickle

def calculate_kl_divergence(student_logits, teacher_logits):
    student_log_probs = nn.functional.log_softmax(student_logits, dim=-1)
    teacher_probs = nn.functional.softmax(teacher_logits, dim=-1)
    kl_div = nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return kl_div

def teacher_tensors_hander(distributions_path, device):
    while True:
        files = sorted([f for f in os.listdir(distributions_path) if f.startswith("distributions_") and f.endswith(".pkl")])
        for file in files:
            file_path = os.path.join(distributions_path, file)
            with open(file_path, 'rb') as f:
                numpy_tensor_list = pickle.load(f)
                for numpy_tensor in numpy_tensor_list:
                    yield torch.tensor(numpy_tensor, device=device)