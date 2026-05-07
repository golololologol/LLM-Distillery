from torch.optim.lr_scheduler import LRScheduler
from torch.optim import Muon
from transformers import get_scheduler
from typing import Any, Protocol, Tuple, cast
import schedulefree
import torch
import math
import re
import os


_BNB_OPTIMIZERS = {
    "adamw": "AdamW",
    "adamw8bit": "AdamW8bit",
    "adamw32bit": "AdamW32bit",
    "paged_adamw": "PagedAdam",
    "paged_adamw8bit": "PagedAdamW8bit",
    "paged_adamw32bit": "PagedAdamW32bit",
    "sgd": "SGD",
    "rmsprop": "RMSprop",
    "rmsprop8bit": "RMSprop8bit",
    "rmsprop32bit": "RMSprop32bit",
    "adagrad": "Adagrad",
}


class _ApolloAdamWConstructor(Protocol):
    def __call__(
        self,
        params: Any,
        *,
        lr: float,
        betas: Tuple[float, float],
        weight_decay: float,
        scale_front: bool,
        no_deprecation_warning: bool,
    ) -> torch.optim.Optimizer: ...


def _bnb_optimizer_class(name: str):
    import contextlib

    with open(os.devnull, 'w') as devnull, contextlib.redirect_stdout(devnull):
        import bitsandbytes as bnb
    return getattr(cast(Any, bnb.optim), _BNB_OPTIMIZERS[name])


def _apollo_adamw_class() -> _ApolloAdamWConstructor:
    from apollo_torch import APOLLOAdamW

    return cast(_ApolloAdamWConstructor, APOLLOAdamW)


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
        target_check = check_target_module_exists(
            target_modules, module_name, return_is_regex=True
        )
        target_module_exists, is_regex = target_check if isinstance(target_check, tuple) else (target_check, False)

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


class CombinedOptimizer(torch.optim.Optimizer):
    def __init__(self, *optimizers):
        self.optimizers = optimizers
        self.param_groups = []
        for opt in optimizers:
            self.param_groups.extend(opt.param_groups)
        self.defaults = {}
        self.state = cast(Any, {})

    def step(self, closure=None):
        for opt in self.optimizers:
            opt.step(closure)

    def zero_grad(self, set_to_none=False):
        for opt in self.optimizers:
            opt.zero_grad(set_to_none=set_to_none)

    def state_dict(self):
        return {f"opt_{i}": opt.state_dict() for i, opt in enumerate(self.optimizers)}

    def load_state_dict(self, state_dict):
        for i, opt in enumerate(self.optimizers):
            opt.load_state_dict(state_dict[f"opt_{i}"])


def set_optimizer(model: Any, lr: float, betas: Tuple[float, float], optimizer_name: str, weight_decay=1e-2, momentum=0.01, nesterov=False, total_steps=200, training_strategy: str = "auto", warmup_steps: int = 0):
    optimizer_name = optimizer_name.lower()
    is_fsdp = training_strategy in ("fsdp2",)
    
    optimizer_mapping = {
        "adamw_torch": torch.optim.AdamW,
        "apollo": None,
        "apollomini": None,
        "schedulefree": schedulefree.AdamWScheduleFree,
        "muon": None,
    }
    optimizer_mapping.update({name: None for name in _BNB_OPTIMIZERS})

    fsdp_upgrades = {
        "adamw": "adamw32bit",
        "adamw8bit": "adamw32bit",
        "paged_adamw8bit": "paged_adamw32bit",
        "rmsprop": "rmsprop32bit",
        "rmsprop8bit": "rmsprop32bit",
    }
    
    if optimizer_name not in optimizer_mapping:
        raise ValueError(
            f"Invalid optimizer name: {optimizer_name}\n"
            f"Available optimizers: {list(optimizer_mapping.keys())}"
        )

    if is_fsdp and optimizer_name in fsdp_upgrades:
        new_name = fsdp_upgrades[optimizer_name]
        print(f"  Warning: Switching {optimizer_name} -> {new_name} for FSDP compatibility (8-bit optimizer states not supported)")
        optimizer_name = new_name

    match optimizer_name:
        case "adamw_torch":
            return torch.optim.AdamW(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8)

        case "schedulefree":
            return schedulefree.AdamWScheduleFree(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, warmup_steps=warmup_steps)

        case "adamw" | "adamw8bit" | "adamw32bit" | "paged_adamw" | "paged_adamw8bit" | "paged_adamw32bit":
            return _bnb_optimizer_class(optimizer_name)(model.parameters(), lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8)
        
        case "sgd":
            return _bnb_optimizer_class(optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
        
        case "rmsprop" | "rmsprop8bit" | "rmsprop32bit":
            return _bnb_optimizer_class(optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay, alpha=0.9, eps=1e-10, centered=True)
        
        case "adagrad":
            return _bnb_optimizer_class(optimizer_name)(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        case "apollo":
            args = {
                'rank': 256,
                'proj': 'random',
                'scale_type': 'channel',
                'scale': 32,
                'update_proj_gap': max(1, total_steps // 10),
                'proj_type': 'std'
            }
            param_groups = setup_low_rank_optimizer(model, optimizer_name, target_modules="all-linear", **args)
            return _apollo_adamw_class()(param_groups, lr=lr, betas=betas, weight_decay=weight_decay, scale_front=True, no_deprecation_warning=True)
                
        case "apollomini":
            args = {
                'rank': 1,
                'proj': 'random',
                'scale_type': 'tensor',
                'scale': 128,
                'update_proj_gap': max(1, total_steps // 10),
                'proj_type': 'std'
            }
            param_groups = setup_low_rank_optimizer(model, optimizer_name, target_modules="all-linear", **args)
            return _apollo_adamw_class()(param_groups, lr=lr, betas=betas, weight_decay=weight_decay, scale_front=True, no_deprecation_warning=True)

        case "muon":
            muon_params = [p for p in model.parameters() if p.ndim >= 2]
            adam_params = [p for p in model.parameters() if p.ndim < 2]
            muon_opt = Muon(muon_params, lr=lr, weight_decay=weight_decay, momentum=momentum, nesterov=nesterov)
            adam_opt = torch.optim.AdamW(adam_params, lr=lr, betas=betas, weight_decay=weight_decay, eps=1e-8)
            return CombinedOptimizer(muon_opt, adam_opt)


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
