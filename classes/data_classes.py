from numpy import ndarray
import hashlib
import torch


class Distribution:
    def __init__(self, origin_convo_id: int, length: int = 0, cropped_end: bool = False, content_ranges: list[tuple[int, int]] = [], tokenized: ndarray = None):
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


class ConvoTokenized:
    def __init__(self, tokenized: ndarray, content_ranges, padding, cropped_end, convo_id):
        self.tokenized: ndarray = tokenized
        self.content_ranges: list[tuple[int, int]] = content_ranges
        self.content_tokens: list[int] = [token for start, end in content_ranges for token in tokenized[start:end]]
        self.padding: int = padding
        self.cropped_end: bool = cropped_end
        self.length: int = len(tokenized) - padding
        self.len_content: int = sum([end - start for start, end in content_ranges])
        self.content_sha: str = hashlib.sha256("".join(map(str, self.content_tokens)).encode()).hexdigest()
        self.origin_convo_id: int = convo_id