import pynvml
import torch

try:
    pynvml.nvmlInit()
except pynvml.NVMLError:
    pass


def get_vram_used() -> list[int]:
    """Returns currently used VRAM in bytes for each GPU, ordered by device index."""
    used = []
    for i in range(num_gpus()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        used.append(info.used)
    return used

def get_vram_free() -> list[int]:
    """Returns currently free VRAM in bytes for each GPU, ordered by device index."""
    free = []
    for i in range(num_gpus()):
        handle = pynvml.nvmlDeviceGetHandleByIndex(i)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        free.append(info.free)
    return free

def num_gpus() -> int:
    """
    Returns the number of available GPUs.
    
    Use this for easier added support for ROCm GPUs in the future"""
    return torch.cuda.device_count()
