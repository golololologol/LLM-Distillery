from torch.optim.lr_scheduler import LRScheduler
from transformers import get_scheduler
from apollo_torch import APOLLOAdamW
from typing import Any, Tuple
from torch import Tensor
import torch.nn.functional as F
import torch
import math
import sys
import re
import os

# shut the the hell up bnb with its `bin ..bitsandbytes\libbitsandbytes_cuda121.dll`
sys.stdout = open(os.devnull, 'w')
import bitsandbytes as bnb
sys.stdout = sys.__stdout__


class WarmupStableDecayLR(LRScheduler):
    """
    Implementation of Warmup Stable Decay learning rate scheduler from MiniCPM.\n
    Has custom cosine warmup instead of linear.\n
    Paper: https://arxiv.org/pdf/2404.06395
    """
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
            cosine = (0.5 * (1 + math.cos(math.pi + progress * math.pi)))
            return [self.constant_lr * cosine for group in self.optimizer.param_groups]
        elif self.warmup_steps <= self.last_epoch <= self.decay_start_step:
            return [self.constant_lr for group in self.optimizer.param_groups]
        elif self.decay_start_step < self.last_epoch < self.total_steps:
            lr_scale = 1 - ((self.last_epoch - self.decay_start_step) / self.decay_steps) * (1 - self.final_lr / self.constant_lr)
            return [self.constant_lr * lr_scale for group in self.optimizer.param_groups]
        else:
            return [self.final_lr for group in self.optimizer.param_groups]
        


def calculate_divergence(student_logits: Tensor, teacher_logits: Tensor, indices: Tensor|None, convo_CE_tokens: Tensor, alpha: Tensor, avoid_indices: Tensor) -> dict[str, Tensor]:
    """
    Custom loss function for distillation with all sorts of shenanigans applied.

    Args:
        student_logits (Tensor): A (num_toks, vocab_size) tensor with the logits from the student model.
        teacher_logits (Tensor): A (num_toks, top_k or vocab_size) tensor with the logits from the teacher model.\\
            The logits are expected to come from the same conversation and start predicting from the same point as the student's logits.
        indices (Tensor | None): A (num_toks, top_k) tensor with the token indices of the top k logits from the teacher model. If None, the logits are used as is.
        convo_CE_tokens (Tensor): A (num_toks) tensor with the token ids from within the content ranges of the conversation.\\
        alpha (Tensor): 
        avoid_indices (Tensor): 

    Returns:
        dict[str,Tensor]:
            A dictionary containing the calculated losses.\\
            It MUST contain the following k/v pair {"train_loss": Tensor} to actually train from!\\
            You may add any other k/v pairs to be used for logging purposes of any other values you want.
    """

    def custom_kl_div(student_logprobs: Tensor, teacher_logprobs: Tensor, per_token: bool = False):
        kl_div_raw = F.kl_div(student_logprobs, teacher_logprobs, reduction='none', log_target=True)

        if per_token:
            kl_div = kl_div_raw.sum(dim=-1)
        else:
            kl_div = kl_div_raw.sum(dim=-1).mean()
        return kl_div

    def abomination_loss(kl_div_per_token: Tensor, alpha: Tensor, CE_loss_per_token: Tensor):
        if CE_loss_per_token.numel() < kl_div_per_token.numel():
            CE_loss_per_token = torch.cat((CE_loss_per_token, torch.zeros(1, device=CE_loss_per_token.device)))

        weights = ((kl_div_per_token / kl_div_per_token.max()) + 1).pow(alpha)
        weights = weights + CE_loss_per_token

        loss = (kl_div_per_token * weights).mean()
        return loss
    
    def custom_ce_loss(logprobs: Tensor, convo_CE_tokens: Tensor, indices: Tensor, ignore_indexes: Tensor) -> Tensor:
        crop = 0
        if convo_CE_tokens.size(-1) < logprobs.size(0):
            crop = -1

        topk_mask = convo_CE_tokens.unsqueeze(-1) == indices[:crop]
        topk_mask[ignore_indexes] = False
        valid_mask = topk_mask.any(dim=-1)
        
        valid_logprobs = torch.zeros_like(valid_mask, dtype=torch.float32)
        valid_logprobs[valid_mask] = logprobs[:crop][topk_mask].squeeze(-1)

        # compute the average of only valid tokens (e.g. tokens that aren't predicting prompt formatting tokens)
        CE_valid_mean = valid_logprobs[valid_mask].mean()

        valid_logprobs[~valid_mask] = CE_valid_mean
        return -valid_logprobs


    min_len = min(student_logits.size(0), teacher_logits.size(0), indices.size(0) if indices is not None else student_logits.size(0))
    student_logprobs = F.log_softmax(student_logits[:min_len], dim=-1)

    if indices is not None:
        teacher_logprobs = F.log_softmax(teacher_logits[:min_len], dim=-1)
        student_logprobs = torch.gather(student_logprobs, 1, indices[:min_len])
    else:
        teacher_logprobs = F.log_softmax(teacher_logits[:min_len], dim=-1)

    convo_CE_tokens = convo_CE_tokens[:min_len]
    CE_loss = custom_ce_loss(student_logprobs, convo_CE_tokens, indices, avoid_indices)
    teacher_CE_loss = custom_ce_loss(teacher_logprobs, convo_CE_tokens, indices, avoid_indices)
        
    CE_diff = CE_loss - teacher_CE_loss

    corrected_CE_diff = torch.where(CE_diff < 0, torch.zeros_like(CE_diff), CE_diff)

    kl_div = custom_kl_div(student_logprobs, teacher_logprobs, per_token=True)
    reverse_kl_div = custom_kl_div(teacher_logprobs, F.log_softmax(student_logprobs, dim=-1), per_token=True)

    weighted_kl_div = abomination_loss(kl_div, alpha, corrected_CE_diff)
    weighted_r_kl_div = abomination_loss(reverse_kl_div, alpha, corrected_CE_diff)

    custom_loss = (weighted_kl_div + weighted_r_kl_div) / 2

    loss_dict = {
        "train_loss": custom_loss,
        "custom loss": custom_loss,
        "CE loss": CE_loss.mean(),
        "kl_div": kl_div.mean(),
        "reverse kl_div": reverse_kl_div.mean(),
        "weighted kl_div": weighted_kl_div,
        "weighted rev. kl_div": weighted_r_kl_div,
        "teacher CE loss": teacher_CE_loss.mean(),
        "CE diff": CE_diff.mean(),
    }

    return loss_dict

def check_target_module_exists(optim_target_modules, key: str, return_is_regex: bool = False):
    # Fully borrowed from here: https://github.com/zhuhanqing/transformers/blob/apollo-integration/src/transformers/trainer_utils.py#L851
    """A helper method to check if the passed module's key name matches any of the target modules in the optim_target_modules.

    Args:
        optim_target_modules (`Union[str, List[str]]`):
            A list of strings to try to match. Can be also a full string.
        key (`str`):
            A key to search any matches in optim_target_modules
        return_is_regex (`bool`):
            If set to `True`, the method will return whether the passed `optim_target_modules`
            is a regex or not.

    Returns:
        `bool` : True of match object if key matches any target modules from config, False or
        None if no match found
        `bool` : If the matched target module is a regex to silence out the warnings in Trainer
        for extra modules being found (only if `target_module_found=True` for an array of regex).
    """
    target_module_found = False
    is_regex = False

    if isinstance(optim_target_modules, str):
        target_module_found = bool(re.fullmatch(optim_target_modules, key))
        is_regex = True if not optim_target_modules == key else False
    elif key in optim_target_modules:  # from here, target_module_found must be a list of str
        # this module is specified directly in target_modules
        target_module_found = True
    elif any(target_key in key for target_key in optim_target_modules):
        target_module_found = True
    elif any(bool(re.fullmatch(optim_target_module, key)) for optim_target_module in optim_target_modules):
        target_module_found = True
        is_regex = True

    if return_is_regex:
        return target_module_found, is_regex

    return target_module_found

def setup_low_rank_optimizer(model, optimizer_name: str, target_modules: list[str]|str, **optim_kwargs) -> list[dict[Any, Any]]:
    # Borrowed from here: https://github.com/zhuhanqing/transformers/blob/apollo-integration/src/transformers/trainer.py#L1315
    # And customized to a certain degree.
    """
    Helper function to set up low-rank optimizers like GaLore and Apollo.
    
    Args:
        model (Any): The model to be optimized.
        optimizer_name (str): The name of the optimizer to be used.
        target_modules (list[str]|str): A list of strings or a regex string to match the target modules.
        optim_kwargs (dict): Additional keyword arguments for the optimizer.
        
    Returns:
        list[dict[Any, Any]]: List of parameter groups for the optimizer.
    """

    if target_modules is None:
        raise ValueError(f"You need to define `target_modules` to use {optimizer_name} optimizers")

    if not isinstance(target_modules, (list, str)):
        raise ValueError(
            f"`target_modules` must be a list of strings, a regex string, or 'all-linear'. Got: {target_modules}"
        )

    if model is None:
        raise ValueError(f"You need to pass a model to initialize {optimizer_name} optimizer.")

    all_linear = (
        isinstance(target_modules, str)
        and target_modules.replace("_", "-").lower() == "all-linear"
    )

    target_params = []
    target_params_names = []
    for module_name, module in model.named_modules():
        target_module_exists, is_regex = check_target_module_exists(
            target_modules, module_name, return_is_regex=True
        )

        if not isinstance(module, torch.nn.Linear):
            if target_module_exists and not is_regex:
                print(
                    f"{module_name} matched but ignored. {optimizer_name} only supports linear layers."
                )
            continue

        if not target_module_exists and not all_linear:
            continue

        target_params.append(module.weight)
        target_params_names.append(module_name + ".weight")

    if len(target_params) == 0:
        raise ValueError(f"No target modules found for {optimizer_name} ({target_modules}).")

    non_target_params = [p for n, p in model.named_parameters() if n not in target_params_names]

    param_groups = [
        {"params": non_target_params},
        {"params": target_params, **optim_kwargs},
    ]
    return param_groups


def set_optimizer(model: Any, lr: float, betas: Tuple[float, float], optimizer_name: str, weight_decay=1e-2, momentum=0.01, nesterov=False):
    """
    A function to set the optimizer for the model.

    Args:
        model (Any): The model to be optimized.
        lr (float): Learning rate to be used.
        betas (tuple(float, float)): A tuple of two floats for the betas of adam optimizers.
        optimizer_name (str): The name of the optimizer to be used.
        weight_decay (float, optional): The weight decay for adam optimizers. Defaults to 1e-2.
        momentum (float, optional): The momentum for SGD optimizer. Defaults to 0.01.
        nesterov (bool, optional): Whether to use nesterov momentum for SGD optimizer. Defaults to False.

    Raises:
        ValueError: If the optimizer name is not valid.

    Returns:
        Optimizer: The optimizer to be used for the model.
    """
    
    optimizer_name = optimizer_name.lower()
    
    optimizer_mapping = {
        "adam": bnb.optim.Adam,
        "adamw": bnb.optim.AdamW,
        "adamw8bit": bnb.optim.AdamW8bit,
        "adamw32bit": bnb.optim.AdamW32bit, 
        "paged_adamw": bnb.optim.PagedAdam,
        "paged_adamw8bit": bnb.optim.PagedAdamW8bit,
        "paged_adamw32bit": bnb.optim.PagedAdamW32bit,
        "sgd": bnb.optim.SGD,
        "rmsprop": bnb.optim.RMSprop,
        "rmsprop8bit": bnb.optim.RMSprop8bit,
        "rmsprop32bit": bnb.optim.RMSprop32bit,
        "adagrad": bnb.optim.Adagrad,
        "apollo": APOLLOAdamW,
        "apollomini": APOLLOAdamW
    }
    
    if optimizer_name not in optimizer_mapping:
        raise ValueError(
            f"Invalid optimizer name: {optimizer_name}\n"
            f"Available optimizers: {list(optimizer_mapping.keys())}"
            )
        
    if optimizer_name in ["adam", "adamw", "adamw8bit", "adamw32bit", "paged_adamw", "paged_adamw8bit", "paged_adamw32bit"]:
        return optimizer_mapping[optimizer_name](model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8)
    
    elif optimizer_name in ["sgd"]:
        return optimizer_mapping[optimizer_name](model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
    
    elif optimizer_name in ["rmsprop", "rmsprop8bit", "rmsprop32bit"]:
        return optimizer_mapping[optimizer_name](model.parameters(), lr=lr, weight_decay=weight_decay, alpha=0.9, eps=1e-10, centered=True)
    
    elif optimizer_name in ["adagrad"]:
        return optimizer_mapping[optimizer_name](model.parameters(), lr=lr, weight_decay=weight_decay)
    
    elif optimizer_name == "apollo":

        args = {
            'rank': 256,
            'proj': 'random',
            'scale_type': 'channel',
            'scale': 1,
            'update_proj_gap': 200,
            'proj_type': 'std'
        }
            
        param_groups = setup_low_rank_optimizer(model, optimizer_name, target_modules="all-linear", **args)

        return optimizer_mapping[optimizer_name](param_groups, lr=lr, betas=betas, weight_decay=weight_decay, scale_front=True, no_deprecation_warning=True)
            
    elif optimizer_name == "apollomini":
        
        args = {
            'rank': 1,
            'proj': 'svd',
            'scale_type': 'tensor',
            'scale': 128,
            'update_proj_gap': 200,
            'proj_type': 'std'
        }
        
        param_groups = setup_low_rank_optimizer(model, optimizer_name, target_modules="all-linear", **args)
            
        return optimizer_mapping[optimizer_name](param_groups, lr=lr, betas=betas, weight_decay=weight_decay, scale_front=True, no_deprecation_warning=True)
    

def set_lr_scheduler(optimizer, lr_scheduler_name: str, num_warmup_steps, num_training_steps, num_epoch_steps, decay_start=0.5, constant_lr=5e-5, final_lr=1e-9):
    lr_scheduler_name = lr_scheduler_name.lower()
    
    lr_scheduler_classes = {
        "wsd": WarmupStableDecayLR
    }
    
    if lr_scheduler_name in lr_scheduler_classes:
        if lr_scheduler_name == "wsd":
            return lr_scheduler_classes[lr_scheduler_name](optimizer, num_training_steps, num_warmup_steps, decay_start, final_lr)
    else:
        return get_scheduler(lr_scheduler_name, optimizer, num_warmup_steps, num_training_steps)
        
