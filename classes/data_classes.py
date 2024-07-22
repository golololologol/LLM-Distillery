from multiprocessing import shared_memory
from numpy import ndarray
import hashlib
import torch


class Distribution:
    """
    Class for probability distribution data.
    """
    def __init__(self, origin_convo_id: int, length: int = 0, cropped_end: bool = False, content_ranges: list[tuple[int, int]] = [], tokenized: ndarray = None, content_sha: str = "", sample: str = ""):
        self.distribution: ndarray|torch.Tensor = None
        self.length: int = length
        self.tokenized: ndarray = tokenized
        self.origin_convo_id: int = origin_convo_id
        self.ppl: float = -1
        self.cropped_end: bool = cropped_end
        self.content_ranges: list[tuple[int, int]] = content_ranges
        self.shd_mem_name: str = ""
        self.distr_shape = None
        self.distr_dtype = None
        self.indices = None
        self.content_sha: str = content_sha
        self.sample: str = sample

    def to_shd_mem(self) -> shared_memory.SharedMemory:
        """
        Put the distribution into shared memory.
        """
        shd_mem = shared_memory.SharedMemory(create=True, size=self.distribution.nbytes)
        self.shd_mem_name = shd_mem.name
        self.distr_shape = self.distribution.shape
        self.distr_dtype = self.distribution.dtype
        shd_distr = ndarray(self.distr_shape, dtype=self.distr_dtype, buffer=shd_mem.buf)
        shd_distr[:] = self.distribution
        del self.distribution
        return shd_mem

    def from_shd_mem(self) -> shared_memory.SharedMemory:
        """
        Retrieve the distribution from shared memory.
        """
        shd_mem = shared_memory.SharedMemory(name=self.shd_mem_name)
        self.distribution = ndarray(self.distr_shape, dtype=self.distr_dtype, buffer=shd_mem.buf)
        return shd_mem

class ConvoTokenized:
    """
    Class for tokenized conversation data.
    """
    def __init__(self, tokenized: ndarray, content_ranges, padding, cropped_end, convo_id, sample: str = ""):
        self.tokenized: ndarray = tokenized
        self.content_ranges: list[tuple[int, int]] = content_ranges
        self.padding: int = padding
        self.cropped_end: bool = cropped_end
        self.length: int = len(tokenized) - padding
        self.len_content: int = sum([end - start for start, end in content_ranges])
        sha_content_tokens = ""
        for start, end in content_ranges:
            sha_content_tokens += "".join(map(str, tokenized[start+1:end]))
        self.content_sha: str = hashlib.sha256(sha_content_tokens.encode()).hexdigest()
        self.origin_convo_id: int = convo_id
        self.sample: str = sample