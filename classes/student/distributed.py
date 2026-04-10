from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist
import torch
import socket
import sys
import os


def find_free_port():
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(('', 0))
        return s.getsockname()[1]


def setup_distributed(rank, world_size):
    if sys.platform == "win32":
        os.environ["USE_LIBUV"] = "0"
    backend = "nccl" if dist.is_nccl_available() else "gloo"
    dist.init_process_group(backend, rank=rank, world_size=world_size)


def get_transformer_layers(model):
    no_split = getattr(model, "_no_split_modules", None)
    if not no_split:
        raise ValueError(f"Model {type(model).__name__} does not define _no_split_modules")

    layer_classes = set()
    for cls_name in no_split:
        for m in model.modules():
            if m.__class__.__name__ == cls_name:
                layer_classes.add(m.__class__)
                break

    if not layer_classes:
        raise ValueError(f"Could not find any modules matching _no_split_modules: {no_split}")

    return [m for m in model.modules() if isinstance(m, tuple(layer_classes))]


def wrap_distributed(model, strategy, training_precision, rank, world_size):
    if strategy == "fsdp2" and training_precision in ("4bit", "8bit"):
        raise ValueError("FSDP2 is incompatible with BitsAndBytes quantization (4bit/8bit). Use DDP or naive_layer_split strategy instead.")
    if strategy == "fsdp2" and not dist.is_nccl_available():
        raise ValueError(
            "FSDP2 requires the NCCL backend, which is not available on this system. "
            "Use 'ddp' or 'naive_layer_split' strategy instead."
        )
    if strategy == "ddp" and world_size > 1:
        return DDP(model, device_ids=[rank], static_graph=True)
    elif strategy == "fsdp2":
        from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16, reduce_dtype=torch.float32)
        for layer in get_transformer_layers(model):
            fully_shard(layer, mp_policy=mp_policy)
        fully_shard(model, mp_policy=mp_policy)
    return model

