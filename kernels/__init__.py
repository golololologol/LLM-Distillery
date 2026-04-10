from kernels.inference import get_inference_kernel
from kernels.skew_kl import (
    fused_skew_kl_forward_backward,
    fused_train_forward_backward_skew_kl,
)
from kernels.akl import (
    fused_akl_forward_backward,
    fused_train_forward_backward_akl,
)
from kernels.abomination import (
    fused_abomination_forward_backward,
    fused_train_forward_backward_abomination,
)


def fused_train_forward_backward(logits, token_ids, byte_vocab, teacher_dists_flat, actual_bytes_flat,
                                  teacher_offsets, target_byte_lens, alpha, loss_type, **kwargs):
    if loss_type == "skew_kl":
        return fused_train_forward_backward_skew_kl(
            logits, token_ids, byte_vocab, teacher_dists_flat, actual_bytes_flat,
            teacher_offsets, target_byte_lens, alpha, **kwargs)
    if loss_type == "akl":
        return fused_train_forward_backward_akl(
            logits, token_ids, byte_vocab, teacher_dists_flat, actual_bytes_flat,
            teacher_offsets, target_byte_lens, alpha)
    if loss_type == "abomination":
        return fused_train_forward_backward_abomination(
            logits, token_ids, byte_vocab, teacher_dists_flat, actual_bytes_flat,
            teacher_offsets, target_byte_lens, alpha)
    return None, None


_KERNEL_GETTERS = {
    "skew_kl": ("kernels.skew_kl", "get_fused_train_skew_kl_kernel"),
    "abomination": ("kernels.abomination", "get_fused_train_abomination_kernel"),
    "akl": ("kernels.akl", "get_fused_train_akl_kernel"),
}


def precompile_training_kernel(loss_type: str):
    entry = _KERNEL_GETTERS.get(loss_type)
    if entry:
        print("  Pre-compiling CUDA kernels...")
        mod = __import__(entry[0], fromlist=[entry[1]])
        getattr(mod, entry[1])()
