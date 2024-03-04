import numpy as np
import torch.nn.functional as F
import torch
import torch.optim.lr_scheduler as lr
from transformers import get_scheduler
import subprocess
import os
import bitsandbytes as bnb
import math
import torch
from torch.optim.lr_scheduler import _LRScheduler

class WarmupStableDecayLR(_LRScheduler):
    def __init__(self, optimizer, total_steps, warmup_steps, decay_start_percentage, final_lr, last_epoch=-1, verbose=False):
        self.total_steps = total_steps
        self.warmup_steps = warmup_steps
        self.decay_start_step = int(total_steps * decay_start_percentage)
        self.decay_steps = self.total_steps - self.decay_start_step
        self.final_lr = final_lr
        self.constant_lr = optimizer.param_groups[0]['lr']

        super().__init__(optimizer, last_epoch, verbose)
    
    def get_lr(self):
        if self.last_epoch < self.warmup_steps:
            progress = self.last_epoch / self.warmup_steps
            cosine = 0.5 * (1 + math.cos(math.pi + progress * math.pi))
            return [self.constant_lr * cosine for group in self.optimizer.param_groups]
        elif self.warmup_steps <= self.last_epoch <= self.decay_start_step:
            return [self.constant_lr for group in self.optimizer.param_groups]
        elif self.decay_start_step < self.last_epoch < self.total_steps:
            lr_scale = 1 - ((self.last_epoch - self.decay_start_step) / self.decay_steps) * (1 - self.final_lr / self.constant_lr)
            return [self.constant_lr * lr_scale for group in self.optimizer.param_groups]
        else:
            return [self.final_lr for group in self.optimizer.param_groups]

    def _get_closed_form_lr(self):
        if self.last_epoch < self.warmup_steps:
            progress = self.last_epoch / self.warmup_steps
            cosine = 0.5 * (1 + math.cos(math.pi + progress * math.pi))
            return [self.constant_lr * cosine for group in self.optimizer.param_groups]
        elif self.warmup_steps <= self.last_epoch <= self.decay_start_step:
            return [self.constant_lr for group in self.optimizer.param_groups]
        elif self.decay_start_step < self.last_epoch < self.total_steps:
            lr_scale = 1 - ((self.last_epoch - self.decay_start_step) / self.decay_steps) * (1 - self.final_lr / self.constant_lr)
            return [self.constant_lr * lr_scale for group in self.optimizer.param_groups]
        else:
            return [self.final_lr for group in self.optimizer.param_groups]


class Params:
    def __init__(self):
        self.model_path = ""
        self.model_name = ""
        self.save_folder = ""
        self.dataset_tokenized = []
        self.dataset_content_ranges = []
        self.validation_dataset_tokenized = []
        self.validation_dataset_content_ranges = []
        self.distributions_path = ""
        self.validation_distributions_path = ""
        self.empty_convo_ids = []
        self.validation_empty_convo_ids = []
        self.validation_per_steps = 0
        self.training_type = ""
        self.save_interval = 0.0
        self.load_in_half = False
        self.shuffle_data = False
        self.optimizer_name = ""
        self.context_length = 0
        self.stop_at_convo = None
        self.full_dataset_len = 0
        self.grad_accum_steps = 0
        self.num_epochs = 0
        self.num_warmup_steps = 0
        self.student_metadata = {}
        self.lr = 0.0
        self.temperature = 0.0
        self.decay_start = 0.0
        self.lr_scheduler_name = ""
        self.custom_reduction = False
        self.device = ""
        self.crop_to_size = 0

def launch_tensorboard(log_dir):
    tensorboard = subprocess.Popen(['tensorboard', '--logdir', os.path.join(log_dir, "tensorboard_logs"), '--bind_all'], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return tensorboard

def calculate_divergence(student_logits, teacher_log_probs: torch.Tensor, temp=1.0, custom=False):
    student_log_probs_temp = F.log_softmax(F.log_softmax(student_logits, dim=-1).clip(-103, 88) / temp, dim=-1)
    teacher_log_probs_temp = F.log_softmax(F.log_softmax(teacher_log_probs, dim=-1).clip(-103, 88) / temp, dim=-1)

    reduction = 'batchmean' if not custom else 'none'
    kl_div = F.kl_div(student_log_probs_temp, teacher_log_probs_temp, reduction=reduction, log_target=True)
    if custom:
        kl_div = ((kl_div.exp() - 1).sum(dim=-1) + 1).mean().log()

    return kl_div

def scale_temperature(current_step, num_training_steps, temperature):
    temp_scaling_steps = num_training_steps / 2
    if current_step < temp_scaling_steps and temperature > 1:
        decay_rate = -np.log(1e-2 / (temperature - 1)) / temp_scaling_steps
        return 1 + (temperature - 1) * np.exp(-decay_rate * current_step)
    else:
        return 1

def set_optimizer(model_parameters, lr, grad_accum_steps, betas, optimizer_name: str, weight_decay=1e-2, momentum=0.9, nesterov=True):
    optimizer_name = optimizer_name.lower()
    
    optimizer_classes = {
        "adam": bnb.optim.Adam,
        "adamw": bnb.optim.AdamW,
        "adamw8bit": bnb.optim.AdamW8bit,
        "adamw32bit": bnb.optim.AdamW32bit, 
        "paged_adamw": bnb.optim.PagedAdam,
        "paged_adamw8bit": bnb.optim.PagedAdamW8bit,
        "paged_adamw32bit": bnb.optim.PagedAdamW32bit,
        "sgd": torch.optim.SGD,
        "rmsprop32bit": bnb.optim.RMSprop32bit,
    }

    if optimizer_name in optimizer_classes:
        if optimizer_name in ["adam", "adamw", "adamw8bit", "adamw32bit", "paged_adamw", "paged_adamw8bit", "paged_adamw32bit"]:
            return optimizer_classes[optimizer_name](model_parameters, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8)
        elif optimizer_name in ["sgd"]:
            return optimizer_classes[optimizer_name](model_parameters, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
        elif optimizer_name in ["rmsprop32bit"]:
            return optimizer_classes[optimizer_name](model_parameters, lr=lr, weight_decay=weight_decay, alpha=0.9, eps=1e-10, centered=True)
    else:
        raise ValueError(f"Invalid optimizer name: {optimizer_name}\nAvailable optimizers: {list(optimizer_classes.keys())}")
    
def set_lr_scheduler(optimizer, lr_scheduler_name, num_warmup_steps, num_training_steps, num_epoch_steps, decay_start=0.5, constant_lr=5e-5, final_lr=5e-7):
    lr_scheduler_name = lr_scheduler_name.lower()
    
    lr_scheduler_classes = {
        "cosine_anneal": lr.CosineAnnealingLR,
        "cosine_anneal_warm": lr.CosineAnnealingWarmRestarts,
        "one_cycle": lr.OneCycleLR,
        "wsd": WarmupStableDecayLR
    }
    
    if lr_scheduler_name in lr_scheduler_classes:
        if lr_scheduler_name == "cosine_anneal":
            return lr_scheduler_classes[lr_scheduler_name](optimizer, T_0=num_epoch_steps, eta_min=5e-7)
        elif lr_scheduler_name == "one_cycle":
            return lr_scheduler_classes[lr_scheduler_name](optimizer, max_lr=5e-5, total_steps=num_training_steps)
        elif lr_scheduler_name == "wsd":
            return lr_scheduler_classes[lr_scheduler_name](optimizer, num_training_steps, num_warmup_steps, decay_start, final_lr)
    else:
        return get_scheduler(lr_scheduler_name, optimizer, num_warmup_steps, num_training_steps)
        
