from dataclasses import dataclass
from multiprocessing import shared_memory
import numpy as np


@dataclass
class ConvoProcessed:
    origin_convo_id: int
    tokens: np.ndarray
    content_byte_ranges: list[tuple[int, int]]
    content_sha: str
    formatted_text: str
    padding: int
    cropped: bool
    length: int
    actual_bytes: np.ndarray = None


class Distribution:
    def __init__(
        self,
        origin_convo_id: int,
        content_sha: str = "",
        cropped: bool = False,
        distribution: np.ndarray | None = None,
    ):
        self.origin_convo_id = origin_convo_id
        self.content_sha = content_sha
        self.cropped = cropped
        self.distribution = distribution
        self.shd_mem_name: str = ""
        self.distr_shape: tuple | None = None
        self.distr_dtype: np.dtype | None = None

    def to_shd_mem(self) -> shared_memory.SharedMemory:
        """
        Moves the distribution to shared memory and updates the metadata accordingly.
        Incurs one copy from the original numpy array to shared memory.
        """
        shd_mem = shared_memory.SharedMemory(create=True, size=self.distribution.nbytes)
        self.shd_mem_name = shd_mem.name
        self.distr_shape = self.distribution.shape
        self.distr_dtype = self.distribution.dtype
        shd_distr = np.ndarray(self.distr_shape, dtype=self.distr_dtype, buffer=shd_mem.buf)
        shd_distr[:] = self.distribution
        del self.distribution
        self.distribution = None
        return shd_mem

    def to_shd_mem_gpu(self, tensor) -> shared_memory.SharedMemory:
        """
        Moves a PyTorch tensor to shared memory and updates the distribution metadata accordingly.
        Incurrs one copy: GPU -> CPU
        """
        import torch
        _TORCH_TO_NP = {torch.float16: np.float16, torch.float32: np.float32, torch.float64: np.float64}
        self.distr_shape = tensor.shape
        self.distr_dtype = _TORCH_TO_NP[tensor.dtype]
        shm = shared_memory.SharedMemory(
            create=True, size=tensor.nelement() * tensor.element_size()
        )
        torch.from_numpy(np.ndarray(self.distr_shape, dtype=self.distr_dtype, buffer=shm.buf)).copy_(tensor)
        self.shd_mem_name = shm.name
        self.distribution = None
        return shm

    def from_shd_mem(self) -> shared_memory.SharedMemory:
        """
        Maps the shared memory back to a numpy array and assigns it to self.distribution.
        Incurrs zero copies.
        """
        shd_mem = shared_memory.SharedMemory(name=self.shd_mem_name)
        self.distribution = np.ndarray(self.distr_shape, dtype=self.distr_dtype, buffer=shd_mem.buf)
        return shd_mem


class BatchResult:
    
    def __init__(self, shm_name, shape, dtype, padding):
        self._shm = shared_memory.SharedMemory(name=shm_name)
        self.data = np.ndarray(shape, dtype=dtype, buffer=self._shm.buf)
        self.padding = padding

    def close(self):
        self._shm.close()
        self._shm.unlink()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


@dataclass
class TrainingState:
    num_trained: int
    epoch: int
    next_accum: int
    next_val: int
    next_save: int
    next_state_save: int | None
    wandb_run_id: str | None = None
    best_val_loss: float | None = None
