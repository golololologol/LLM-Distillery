"""ExLlamaV2 inference worker - runs in a spawned process."""
import signal
import torch
import numpy as np


def _primary_device_idx(device_idx) -> int:
    if isinstance(device_idx, list):
        return int(device_idx[0])
    return int(device_idx)


def _compute_vram_budget(config, gpu_split, batch_size, context_len, tokenizer_name):
    """Adjust gpu_split to reserve room for KV cache, byte vocab index, and inference overhead.
    Only used for multi-GPU splitting where workers share GPUs."""
    from exllamav2.model import ExLlamaV2
    from exllamav2.attn import ExLlamaV2Attention
    from transformers import AutoTokenizer
    from classes.byte_vocab import ByteVocabIndex

    gpu_split = list(gpu_split)
    kv_per_layer = batch_size * context_len * config.num_key_value_heads * config.head_dim * 4
    logits_bytes = batch_size * context_len * config.vocab_size * 2
    allocator_margin = 256 * 1024**2

    probe = ExLlamaV2(config)
    probe.set_device_map(gpu_split)
    head_gpu = _primary_device_idx(probe.modules[-1].device_idx)
    scratch = max(m.scratch_space_fixed() for m in probe.modules)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    pre = torch.cuda.memory_allocated(head_gpu)
    byte_vocab = ByteVocabIndex(tokenizer, device=f"cuda:{head_gpu}")
    byte_vocab_bytes = torch.cuda.memory_allocated(head_gpu) - pre

    for i in range(len(gpu_split)):
        num_attention_layers = sum(isinstance(m, ExLlamaV2Attention) and m.device_idx == i for m in probe.modules)
        gpu_split[i] = max(0, gpu_split[i] - (num_attention_layers * kv_per_layer + (byte_vocab_bytes if i == head_gpu else 0)) / 1024**3)
    deferred = {head_gpu: logits_bytes + scratch + allocator_margin}

    del probe
    return gpu_split, deferred, byte_vocab


def inference_worker(
    inference_queue,
    result_queue,
    distribution_writer,
    load_status_queue,
    model_path: str,
    context_len: int,
    batch_size: int,
    gpu_split: list[float],
    seq_chunk_len: int = 256,
    marg_chunk_size: int = 256,
    tokenizer_name: str = "",
    temperature: float = 1.0,
    specials_config: dict | None = None,
    event_extensions: list | None = None,
    target_gpu: int = -1,
):
    def signal_handler(signum, frame):
        while not inference_queue.empty():
            try:
                inference_queue.get_nowait()
            except Exception:
                break
        exit(0)

    exclusive = target_gpu >= 0
    if exclusive:
        torch.cuda.set_device(target_gpu)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    from utils.kernel_utils import preload_torch_extensions
    preload_torch_extensions(["exllamav2_ext", "fused_inference_byte_marginalize"])

    try:
        from exllamav2.cache import ExLlamaV2Cache
        from exllamav2.config import ExLlamaV2Config
        from exllamav2.model import ExLlamaV2
        from classes.data_classes import Distribution

        config = ExLlamaV2Config()
        config.model_dir = model_path
        config.prepare()
        config.max_seq_len = context_len
        config.max_batch_size = batch_size
        config.max_input_len = seq_chunk_len
        config.max_attention_size = context_len ** 2

        if exclusive:
            from transformers import AutoTokenizer
            from classes.byte_vocab import ByteVocabIndex
            model = ExLlamaV2(config)
            model.load(gpu_split=gpu_split)
            cache = ExLlamaV2Cache(model, batch_size, context_len)
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            byte_vocab = ByteVocabIndex(tokenizer, device=f"cuda:{target_gpu}")
            deferred_overhead = {}
        else:
            gpu_split, deferred_overhead, byte_vocab = _compute_vram_budget(config, gpu_split, batch_size, context_len, tokenizer_name)
            model = ExLlamaV2(config)
            model.load(gpu_split=gpu_split)
            cache = ExLlamaV2Cache(model, batch_size, context_len)
            # Match current CUDA device to the byte_vocab/model head GPU so
            # custom kernels (byte marginalize, fused losses) launch on the
            # right stream. Otherwise cuda:0 is used and foreign-device reads
            # produce cudaErrorIllegalAddress.
            torch.cuda.set_device(byte_vocab._device)

        # v8 dual-channel attachment (plan §4, §8).
        ev_alphabet = None
        ev_supported_mask = 0
        ev_specials_hash = None
        if specials_config:
            from transformers import AutoTokenizer
            from classes.event_channel import EventChannelSpec
            tok = AutoTokenizer.from_pretrained(tokenizer_name)
            event_spec = EventChannelSpec.from_tokenizer(tok, specials_config, event_extensions or ())
            byte_vocab.attach_specials(event_spec.vocab, event_spec.resolved)  # routes special-token mass into event slots
            ev_alphabet = event_spec.alphabet
            ev_supported_mask = event_spec.supported_mask
            ev_specials_hash = event_spec.specials_hash
    except Exception as e:
        load_status_queue.put(("failed", str(e)))
        torch.cuda.empty_cache()
        return

    load_status_queue.put(("ready", deferred_overhead))

    shared_mems = []

    try:
        while True:
            item = inference_queue.get()
            if item is None:
                break

            try:
                batch_tokens_np, batch_metadata = item

                cache.current_seq_len = 0
                batch_tensor = torch.from_numpy(batch_tokens_np)
                forward_out = model.forward(batch_tensor, cache=cache)
                if forward_out is None:
                    raise RuntimeError("ExLlamaV2 forward returned None")
                if isinstance(forward_out, tuple):
                    forward_out = forward_out[0]
                batch_logits = forward_out.contiguous()

                if temperature != 1.0:
                    batch_logits = batch_logits / temperature

                distributions = []
                for idx, convo_metadata in enumerate(batch_metadata):
                    ranges = convo_metadata.get("content_byte_ranges", [])
                    length = convo_metadata.get("length", batch_logits.shape[1])
                    anchor_positions = convo_metadata.get("anchor_token_positions") or []
                    event_template = convo_metadata.get("event_manifest_template") or []

                    content_dist_gpu, event_dist_gpu = byte_vocab.marginalise_dual(
                        batch_logits[idx], batch_tensor[idx], ranges,
                        anchor_token_positions=anchor_positions,
                        length=length, T_CHUNK=marg_chunk_size,
                    )

                    # A sample with zero content byte ranges produces an empty
                    # distribution ([0, 256] tensor).  Storing it would crash
                    # SharedMemory(size=0) and yields no training signal anyway.
                    if content_dist_gpu.numel() == 0:
                        continue

                    dist = Distribution(
                        origin_convo_id=convo_metadata["id"],
                        content_sha=convo_metadata["content_sha"],
                        cropped=convo_metadata.get("cropped", False),
                    )
                    dist.segment_manifest = convo_metadata.get("segment_manifest")
                    shared_mems.append(dist.to_shd_mem_gpu(content_dist_gpu))

                    if event_dist_gpu is not None and event_dist_gpu.numel() > 0 and event_template:
                        kept_template = []
                        for tpl, pos in zip(event_template, anchor_positions):
                            if 0 <= int(pos) < length:
                                kept_template.append(dict(tpl))
                        for row, entry in enumerate(kept_template):
                            entry["row"] = row
                        dist.event_manifest = kept_template
                        ev_shm = dist.events_to_shd_mem_gpu(event_dist_gpu)
                        if ev_shm is not None:
                            shared_mems.append(ev_shm)
                    else:
                        dist.event_manifest = []
                    dist.event_alphabet = ev_alphabet
                    dist.supported_mask = ev_supported_mask
                    dist.specials_hash = ev_specials_hash
                    distributions.append(dist)

                distribution_writer.write_batch(distributions)
                result_queue.put(len(batch_metadata))

                while len(shared_mems) > 20 * batch_size:
                    old = shared_mems.pop(0)
                    try:
                        old.close()
                        old.unlink()
                    except Exception:
                        pass
            except Exception as e:
                print(f"  WARNING: Inference worker error: {e}")
                break
    finally:
        try:
            model.unload()
            del model, cache
        except NameError:
            pass
        torch.cuda.empty_cache()
        result_queue.put(None)
