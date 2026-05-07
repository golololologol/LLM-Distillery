"""Isolated subprocess runner for fused training kernel tests.
Exit codes: 0=pass, 1=fail, 2=skip"""
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
torch.manual_seed(42)

from transformers import AutoTokenizer
from classes.byte_vocab import ByteVocabIndex
from classes.losses import _skew_kl_loss, _akl_loss, _abomination_loss, _wasserstein_loss, _jsd_loss, _hellinger_loss
from kernels import fused_train_forward_backward

LOSS_FNS = {
    "skew_kl": _skew_kl_loss, "akl": _akl_loss, "abomination": _abomination_loss,
    "wasserstein": _wasserstein_loss, "jsd": _jsd_loss, "hellinger": _hellinger_loss,
}


def make_data(bv, T1):
    V = bv.vocab_size
    T = T1 + 1
    logits = torch.randn(T, V, device="cuda", dtype=torch.float16) * 0.5
    token_ids = torch.randint(0, V, (T,), device="cuda")
    next_byte_lens = bv.token_byte_lens[token_ids[1:]].long()
    offsets = torch.zeros(T1, dtype=torch.long, device="cuda")
    if T1 > 1:
        offsets[1:] = next_byte_lens[:-1].cumsum(0)
    N = int(offsets[-1].item()) + int(next_byte_lens[-1].item())
    td = torch.rand(N, 256, device="cuda", dtype=torch.float32)
    td = td / td.sum(dim=1, keepdim=True)
    ab = torch.randint(0, 256, (N,), device="cuda", dtype=torch.int32)
    return logits, token_ids, td, ab, offsets, next_byte_lens


def run_parity(bv, loss_type):
    loss_fn = LOSS_FNS[loss_type]
    logits, token_ids, td, ab, offsets, nbl = make_data(bv, 8)
    g, l = fused_train_forward_backward(logits, token_ids, bv, td, ab, offsets, nbl, 0.1, loss_type)
    torch.cuda.synchronize()
    if g is None or l is None:
        print("Kernel compilation failed")
        return 2

    logits_ref = logits.clone().detach().requires_grad_(True)
    student = bv.marginalize_train(logits_ref, token_ids, T_CHUNK=512)
    ref_loss = loss_fn(student, td, ab, 0.1)
    ref_loss["train_loss"].backward()
    assert logits_ref.grad is not None
    ref_g = logits_ref.grad.float()[:8]

    for key in ["kl_div", "CE loss", "train_loss"]:
        rv, fv = ref_loss[key].item(), l[key].item()
        rd = abs(rv - fv) / (abs(rv) + 1e-8)
        if rd >= 0.01:
            print(f"FAIL: {key} reldiff={rd:.6f} ref={rv:.6f} fused={fv:.6f}")
            return 1

    cos = torch.nn.functional.cosine_similarity(ref_g.flatten().unsqueeze(0), g.float().flatten().unsqueeze(0)).item()
    if cos < 0.99:
        print(f"FAIL: gradient cosine={cos:.6f}")
        return 1
    return 0


def run_loss_keys(bv):
    logits, token_ids, td, ab, offsets, nbl = make_data(bv, 4)
    _, l = fused_train_forward_backward(logits, token_ids, bv, td, ab, offsets, nbl, 0.1, "skew_kl")
    torch.cuda.synchronize()
    if l is None:
        print("Kernel compilation failed")
        return 2
    for key in ["train_loss", "custom loss", "kl_div", "CE loss"]:
        if key not in l:
            print(f"FAIL: missing key '{key}'")
            return 1
    return 0


if __name__ == "__main__":
    test_name = sys.argv[1]
    tokenizer = AutoTokenizer.from_pretrained("test_data/tiny_tokenizer")
    bv = ByteVocabIndex(tokenizer, device="cuda")

    if test_name == "loss_keys":
        sys.exit(run_loss_keys(bv))
    elif test_name.endswith("_parity"):
        loss_type = test_name.replace("_parity", "")
        sys.exit(run_parity(bv, loss_type))
    else:
        print(f"Unknown test: {test_name}")
        sys.exit(1)
