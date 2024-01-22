import torch.nn as nn
import torch
import gc
import bitsandbytes as bnb
import os
import pickle
import math

def calculate_kl_divergence(student_logits, teacher_probs):
    epsilon = 1e-6
    student_log_probs = nn.functional.log_softmax(student_logits, dim=-1) + epsilon
    kl_div = nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return kl_div

def teacher_tensors_hander(distributions_path, device, loop=True, empty_convo_ids=[]):
    convo_id = 0
    while True:
        files = sorted([f for f in os.listdir(distributions_path) if f.startswith("distributions_") and f.endswith(".pkl")])
        for file in files:
            file_path = os.path.join(distributions_path, file)
            with open(file_path, 'rb') as f:
                numpy_tensor_list = pickle.load(f)
                for numpy_tensor in numpy_tensor_list:
                    if convo_id not in empty_convo_ids:
                        with torch.no_grad():
                            tensor = torch.tensor(numpy_tensor, device=device)
                        yield tensor
                        del tensor
                    convo_id += 1

                gc.collect()

        if not loop:
            break

def preprocess_logits(distributions_tensor: torch.Tensor, mask):
    if mask is not None:
        distributions_tensor = distributions_tensor[:, mask]
    return distributions_tensor

def create_mask(metadata: dict, additional_tokens: list = []):
    added_tokens_ids = metadata.get("added_tokens_ids", [])
    added_tokens_ids.extend(additional_tokens)
    if added_tokens_ids:
        print(f"Ids to be excluded: {added_tokens_ids}")
        mask = ~torch.isin(torch.arange(metadata['vocab_size']), torch.tensor(added_tokens_ids))
        return mask
    return None

def linear_softmax(tensor):
    min_val = torch.min(tensor)
    shifted_tensor = tensor - min_val + 1e-6
    return shifted_tensor / torch.sum(shifted_tensor)

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
        "sgd": torch.optim.SGD,
        "rmsprop32bit": bnb.optim.RMSprop32bit
    }

    lr = lr * math.sqrt(grad_accum_steps)
    if optimizer_name in optimizer_classes:
        optimizer_class = optimizer_classes[optimizer_name]
        return optimizer_class(model_parameters, lr=lr, betas=betas)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}\nAvailable optimizers: {list(optimizer_classes.keys())}")
