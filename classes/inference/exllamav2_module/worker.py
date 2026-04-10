"""ExLlamaV2 inference worker - runs in a spawned process."""
import signal
import torch
import numpy as np


def _compute_vram_budget(config, gpu_split, batch_size, context_len, tokenizer_name):
    """Adjust gpu_split to reserve room for KV cache, byte vocab index, and inference overhead.
    Only used for multi-GPU splitting where workers share GPUs."""
    from exllamav2 import ExLlamaV2
    from exllamav2.attn import ExLlamaV2Attention
    from transformers import AutoTokenizer
    from classes.byte_vocab import ByteVocabIndex

    gpu_split = list(gpu_split)
    kv_per_layer = batch_size * context_len * config.num_key_value_heads * config.head_dim * 4
    logits_bytes = batch_size * context_len * config.vocab_size * 2
    allocator_margin = 256 * 1024**2

    probe = ExLlamaV2(config)
    probe.set_device_map(gpu_split)
    head_gpu = probe.modules[-1].device_idx
    scratch = max(m.scratch_space_fixed() for m in probe.modules)

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    pre = torch.cuda.memory_allocated(head_gpu)
    byte_vocab = ByteVocabIndex(tokenizer, device=f"cuda:{head_gpu}")
    byte_vocab_bytes = torch.cuda.memory_allocated(head_gpu) - pre

    for i in range(len(gpu_split)):
        n_attn = sum(isinstance(m, ExLlamaV2Attention) and m.device_idx == i for m in probe.modules)
        gpu_split[i] = max(0, gpu_split[i] - (n_attn * kv_per_layer + (byte_vocab_bytes if i == head_gpu else 0)) / 1024**3)
    deferred = {head_gpu: logits_bytes + scratch + allocator_margin}

    del probe
    return gpu_split, deferred, byte_vocab


def inference_worker(
    inference_queue,
    result_queue,
    dm_queue,
    load_status_queue,
    model_path: str,
    context_len: int,
    batch_size: int,
    gpu_split: list[float],
    seq_chunk_len: int = 256,
    marg_chunk_size: int = 256,
    tokenizer_name: str = "",
    temperature: float = 1.0,
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
        from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
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
                batch_logits = model.forward(batch_tensor, cache=cache).contiguous()

                if temperature != 1.0:
                    batch_logits = batch_logits / temperature

                distributions = []
                for idx, convo_metadata in enumerate(batch_metadata):
                    ranges = convo_metadata.get("content_byte_ranges", [])
                    length = convo_metadata.get("length", batch_logits.shape[1])

                    content_dist_gpu = byte_vocab.marginalize_content(
                        batch_logits[idx], batch_tensor[idx], ranges,
                        length=length, T_CHUNK=marg_chunk_size
                    )

                    dist = Distribution(
                        origin_convo_id=convo_metadata["id"],
                        content_sha=convo_metadata["content_sha"],
                        cropped=convo_metadata.get("cropped", False),
                    )
                    shared_mems.append(dist.to_shd_mem_gpu(content_dist_gpu))
                    distributions.append(dist)

                dm_queue.put(('put_batch', distributions))
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
