from __future__ import annotations
import importlib.util
import os
import sys

import torch

try:
    from kernels import fused_train_forward_backward
    _fused_train_fn = fused_train_forward_backward
except (ImportError, Exception):
    _fused_train_fn = None


def preload_torch_extensions(names: list[str]):
    """Pre-import cached PyTorch JIT extensions into sys.modules to skip per-process validation."""
    from torch.utils.cpp_extension import _get_build_directory
    ext_suffix = '.pyd' if os.name == 'nt' else '.so'
    for name in names:
        if name in sys.modules:
            continue
        try:
            build_dir = _get_build_directory(name, verbose=False)
            path = os.path.join(build_dir, name + ext_suffix)
            if os.path.exists(path):
                spec = importlib.util.spec_from_file_location(name, path)
                mod = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(mod)
                sys.modules[name] = mod
        except Exception:
            pass


class _FusedLossGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, student_dists, precomputed_grad, loss_value):
        ctx.save_for_backward(precomputed_grad)
        return loss_value.detach().clone()

    @staticmethod
    def backward(ctx, grad_output):
        precomputed_grad, = ctx.saved_tensors
        return precomputed_grad * grad_output, None, None


def _try_fused_train(logits, token_ids, convo, teacher_dists, actual_bytes_t, byte_vocab, alpha, loss_type, **kwargs):
    if _fused_train_fn is None:
        return None

    length = convo.length
    T1 = length - 1
    device = logits.device

    active_idx, active_byte_lens, content_mask = byte_vocab._compute_content_positions(
        token_ids, convo.content_byte_ranges, length, device
    )

    if len(active_idx) == 0:
        return None

    N_total_active = active_byte_lens.sum().item()
    N_content = content_mask.sum().item()

    n = min(N_content, teacher_dists.shape[0])

    if N_content == N_total_active:
        teacher_dists_k = teacher_dists[:n]
        actual_bytes_k = actual_bytes_t[:n].int()
        loss_scale = 1.0
    else:
        teacher_padded = torch.full((N_total_active, 256), 1.0 / 256, dtype=torch.float32, device=device)
        actual_padded = torch.zeros(N_total_active, dtype=torch.int32, device=device)
        content_indices = content_mask.nonzero(as_tuple=True)[0]
        teacher_padded[content_indices[:n]] = teacher_dists[:n]
        actual_padded[content_indices[:n]] = actual_bytes_t[:n].int()
        teacher_dists_k = teacher_padded
        actual_bytes_k = actual_padded
        loss_scale = N_total_active / N_content

    # Build teacher_offsets and target_byte_lens for active positions
    teacher_offsets = torch.zeros(len(active_idx), dtype=torch.long, device=device)
    if len(active_idx) > 1:
        teacher_offsets[1:] = active_byte_lens[:-1].cumsum(0)

    # Build logits/token_ids for the active subset
    if len(active_idx) == T1:
        chunk_logits = logits[:length]
        chunk_token_ids = token_ids[:length]
    else:
        chunk_logits = torch.nn.functional.pad(logits[active_idx], (0, 0, 0, 1))
        chunk_token_ids = torch.cat([token_ids[:1], token_ids[active_idx + 1]])

    grad_logits_chunk, loss_dict = _fused_train_fn(
        chunk_logits, chunk_token_ids, byte_vocab,
        teacher_dists_k, actual_bytes_k, teacher_offsets, active_byte_lens,
        alpha, loss_type, **kwargs,
    )
    if grad_logits_chunk is None:
        return None

    if loss_scale != 1.0:
        for key in loss_dict:
            if isinstance(loss_dict[key], torch.Tensor) and loss_dict[key].dim() == 0:
                loss_dict[key] = loss_dict[key] * loss_scale
        grad_logits_chunk = grad_logits_chunk * loss_scale

    # Scatter gradients back to full logits shape
    if len(active_idx) == T1:
        grad_logits = torch.zeros_like(logits)
        grad_logits[:T1] = grad_logits_chunk.to(grad_logits.dtype)
    else:
        grad_logits = torch.zeros_like(logits)
        grad_logits[active_idx] = grad_logits_chunk.to(grad_logits.dtype)

    loss_dict["train_loss"] = _FusedLossGrad.apply(logits, grad_logits, loss_dict["train_loss"])
    return loss_dict
