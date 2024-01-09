import torch

def preprocess_logits(distributions_tensor: torch.Tensor, device: str, metadata: dict):
    