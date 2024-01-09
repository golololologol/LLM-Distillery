import torch.nn as nn
import torch
import bitsandbytes as bnb
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
    
def set_optimizer(model_parameters, lr, grad_accum_steps, betas, optimizer_name: str):
    optimizer_name = optimizer_name.lower()
    
    optimizer_classes = {
        "adam": torch.optim.Adam,
        "adamw": bnb.optim.AdamW,
        "adamw8bit": bnb.optim.AdamW8bit,
        "adamw32bit": bnb.optim.AdamW32bit, 
        "paged_adamw": bnb.optim.PagedAdam,
        "paged_adamw8bit": bnb.optim.PagedAdamW8bit,
        "paged_adamw32bit": bnb.optim.PagedAdamW32bit,
        "adagrad8bit": bnb.optim.Adagrad8bit,
        "adagrad32bit": bnb.optim.Adagrad32bit,
        "sgd": torch.optim.SGD
    }
    
    lr = lr * (grad_accum_steps^0.5)
    if optimizer_name in optimizer_classes:
        optimizer_class = optimizer_classes[optimizer_name]
        return optimizer_class(model_parameters, lr=lr, betas=betas)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}\nAvailable optimizers: {list(optimizer_classes.keys())}")