from kernels.inference import get_inference_kernel
from kernels.losses.skew_kl import (
    fused_skew_kl_forward_backward,
    fused_train_forward_backward_skew_kl,
)
from kernels.losses.akl import (
    fused_akl_forward_backward,
    fused_train_forward_backward_akl,
)
from kernels.losses.abomination import (
    fused_abomination_forward_backward,
    fused_train_forward_backward_abomination,
)
from kernels.losses.wasserstein import (
    fused_wasserstein_forward_backward,
    fused_train_forward_backward_wasserstein,
)
from kernels.losses.jsd import (
    fused_jsd_forward_backward,
    fused_train_forward_backward_jsd,
)
from kernels.losses.hellinger import (
    fused_hellinger_forward_backward,
    fused_train_forward_backward_hellinger,
)


def fused_train_forward_backward(logits, token_ids, byte_vocab, teacher_dists_flat, actual_bytes_flat,
                                  teacher_offsets, target_byte_lens, alpha, loss_type, **kwargs):
    entropy_weighting = kwargs.pop("entropy_weighting", False)
    byte_mask = kwargs.pop("byte_mask", None)
    if loss_type == "skew_kl":
        return fused_train_forward_backward_skew_kl(
            logits, token_ids, byte_vocab, teacher_dists_flat, actual_bytes_flat,
            teacher_offsets, target_byte_lens, alpha, entropy_weighting=entropy_weighting,
            byte_mask=byte_mask, **kwargs)
    if loss_type == "akl":
        return fused_train_forward_backward_akl(
            logits, token_ids, byte_vocab, teacher_dists_flat, actual_bytes_flat,
            teacher_offsets, target_byte_lens, alpha, entropy_weighting=entropy_weighting,
            byte_mask=byte_mask)
    if loss_type == "abomination":
        return fused_train_forward_backward_abomination(
            logits, token_ids, byte_vocab, teacher_dists_flat, actual_bytes_flat,
            teacher_offsets, target_byte_lens, alpha, entropy_weighting=entropy_weighting,
            byte_mask=byte_mask)
    if loss_type == "wasserstein":
        return fused_train_forward_backward_wasserstein(
            logits, token_ids, byte_vocab, teacher_dists_flat, actual_bytes_flat,
            teacher_offsets, target_byte_lens, alpha, entropy_weighting=entropy_weighting,
            byte_mask=byte_mask)
    if loss_type == "jsd":
        return fused_train_forward_backward_jsd(
            logits, token_ids, byte_vocab, teacher_dists_flat, actual_bytes_flat,
            teacher_offsets, target_byte_lens, alpha, entropy_weighting=entropy_weighting,
            byte_mask=byte_mask)
    if loss_type == "hellinger":
        return fused_train_forward_backward_hellinger(
            logits, token_ids, byte_vocab, teacher_dists_flat, actual_bytes_flat,
            teacher_offsets, target_byte_lens, alpha, entropy_weighting=entropy_weighting,
            byte_mask=byte_mask)
    return None, None


_KERNEL_GETTERS = {
    "skew_kl": ("kernels.losses.skew_kl", "get_fused_train_skew_kl_kernel"),
    "abomination": ("kernels.losses.abomination", "get_fused_train_abomination_kernel"),
    "akl": ("kernels.losses.akl", "get_fused_train_akl_kernel"),
    "wasserstein": ("kernels.losses.wasserstein", "get_fused_train_wasserstein_kernel"),
    "jsd": ("kernels.losses.jsd", "get_fused_train_jsd_kernel"),
    "hellinger": ("kernels.losses.hellinger", "get_fused_train_hellinger_kernel"),
}


def precompile_training_kernel(loss_type: str):
    entry = _KERNEL_GETTERS.get(loss_type)
    if entry:
        print("\n  Pre-compiling CUDA kernels...")
        mod = __import__(entry[0], fromlist=[entry[1]])
        getattr(mod, entry[1])()
