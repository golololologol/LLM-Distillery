import torch.nn as nn
import torch
import bitsandbytes as bnb
import math

def calculate_kl_divergence(student_logits, teacher_probs, temp=1, per_token=False):
    #print(student_logits)
    student_log_probs = nn.functional.log_softmax(student_logits/temp, dim=-1)
    reduction = 'batchmean' if not per_token else 'sum'
    kl_div = nn.functional.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return kl_div

def scale_temperature(current_step, num_training_steps, temperature):
    return temperature - (temperature - 1) * (current_step / num_training_steps)

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
