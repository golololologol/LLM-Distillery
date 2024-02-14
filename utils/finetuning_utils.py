import torch.nn as nn
import torch
import torch.optim.lr_scheduler as lr
from transformers import get_scheduler
import bitsandbytes as bnb
import math

def calculate_kl_divergence(student_logits, teacher_probs, temp=1, per_token=False):
    student_log_probs = nn.functional.log_softmax(student_logits/temp, dim=-1)
    reduction = 'batchmean' if not per_token else 'sum'
    kl_div = nn.functional.kl_div(student_log_probs, teacher_probs, reduction=reduction)
    return kl_div

def scale_temperature(current_step, num_training_steps, temperature):
    return temperature - (temperature - 1) * (current_step / num_training_steps)

def set_optimizer(model_parameters, lr, grad_accum_steps, betas, optimizer_name: str, weight_decay=1e-2, momentum=0.9, nesterov=True):
    optimizer_name = optimizer_name.lower()
    
    optimizer_classes = {
        "adam": torch.optim.Adam,
        "adamw": bnb.optim.AdamW,
        "adamw8bit": bnb.optim.AdamW8bit,
        "adamw32bit": bnb.optim.AdamW32bit, 
        "paged_adamw": bnb.optim.PagedAdam,
        "paged_adamw8bit": bnb.optim.PagedAdamW8bit,
        "paged_adamw32bit": bnb.optim.PagedAdamW32bit,
        "sgd": torch.optim.SGD,
        "rmsprop32bit": bnb.optim.RMSprop32bit
    }

    lr = lr * math.sqrt(grad_accum_steps)
    if optimizer_name in optimizer_classes:
        if optimizer_name in ["adam", "adamw", "adamw8bit", "adamw32bit", "paged_adamw", "paged_adamw8bit", "paged_adamw32bit"]:
            return optimizer_classes[optimizer_name](model_parameters, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8)
        elif optimizer_name in ["sgd"]:
            return optimizer_classes[optimizer_name](model_parameters, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
        elif optimizer_name in ["rmsprop32bit"]:
            return optimizer_classes[optimizer_name](model_parameters, lr=lr, weight_decay=weight_decay, alpha=0.9, eps=1e-10, centered=True)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}\nAvailable optimizers: {list(optimizer_classes.keys())}")
    
def set_lr_scheduler(optimizer, lr_scheduler_name, num_warmup_steps, num_training_steps, num_epoch_steps):
    lr_scheduler_name = lr_scheduler_name.lower()
    
    lr_scheduler_classes = {
        "cosine": lr.CosineAnnealingLR,
        "step": lr.StepLR,
        "cosine_anneal": lr.CosineAnnealingWarmRestarts,
        "one_cycle": lr.OneCycleLR
    }
    
    if lr_scheduler_name in lr_scheduler_classes:
        if lr_scheduler_name == "cosine":
            return lr_scheduler_classes[lr_scheduler_name](optimizer, T_max=num_training_steps)
        elif lr_scheduler_name == "step":
            return lr_scheduler_classes[lr_scheduler_name](optimizer, step_size=num_epoch_steps, gamma=0.5)
        elif lr_scheduler_name == "cosine_anneal":
            return lr_scheduler_classes[lr_scheduler_name](optimizer, T_0=num_epoch_steps, eta_min=5e-7)
        elif lr_scheduler_name == "one_cycle":
            return lr_scheduler_classes[lr_scheduler_name](optimizer, max_lr=5e-5, total_steps=num_training_steps)
    else:
        return get_scheduler(lr_scheduler_name, optimizer, num_warmup_steps, num_training_steps)
        
