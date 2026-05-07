"""ExLlamaV3 inference worker - runs in a spawned process."""
import signal
import torch
import numpy as np


def _compute_vram_budget(config, gpu_split, batch_size, context_len, tokenizer_name):
    """Adjust gpu_split to reserve room for logits tensor, byte vocab index, and inference overhead.
    Only used for multi-GPU splitting where workers share GPUs."""
    from transformers import AutoTokenizer
    from classes.byte_vocab import ByteVocabIndex

    gpu_split = list(gpu_split)
    logits_bytes = batch_size * context_len * config.vocab_size * 2
    allocator_margin = 256 * 1024**2

    # Head GPU is the last one with nonzero allocation
    head_gpu = 0
    for i in range(len(gpu_split)):
        if gpu_split[i] > 0:
            head_gpu = i

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    pre = torch.cuda.memory_allocated(head_gpu)
    byte_vocab = ByteVocabIndex(tokenizer, device=f"cuda:{head_gpu}")
    byte_vocab_bytes = torch.cuda.memory_allocated(head_gpu) - pre

    # Reserve room for both byte_vocab AND the full-sequence logits tensor.
    # Logits shape is [batch_size, context_len, vocab_size] fp16 — computed
    # AFTER model.load() so they aren't accounted for by use_per_device.
    head_reserved = byte_vocab_bytes + logits_bytes + allocator_margin
    for i in range(len(gpu_split)):
        if i == head_gpu:
            gpu_split[i] = max(0, gpu_split[i] - head_reserved / 1024**3)
    deferred = {head_gpu: logits_bytes + allocator_margin}

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
    preload_torch_extensions(["fused_inference_byte_marginalize"])

    try:
        from exllamav3 import Config, Model
        from classes.data_classes import Distribution

        config = Config.from_directory(model_path)
        config.override_dynamic_seq_len(context_len)

        model = Model.from_config(config)

        if exclusive:
            from transformers import AutoTokenizer
            from classes.byte_vocab import ByteVocabIndex
            model.load(
                device=f"cuda:{target_gpu}",
                max_batch_size=batch_size,
                max_output_size=batch_size * context_len,
            )
            tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            byte_vocab = ByteVocabIndex(tokenizer, device=f"cuda:{target_gpu}")
            deferred_overhead = {}
        else:
            gpu_split, deferred_overhead, byte_vocab = _compute_vram_budget(config, gpu_split, batch_size, context_len, tokenizer_name)
            model.load(
                use_per_device=gpu_split,
                max_batch_size=batch_size,
                max_output_size=batch_size * context_len,  # full-sequence logits, not single-token
            )
            # Custom CUDA kernels (byte marginalize, fused losses) call
            # ``at::cuda::getCurrentCUDAStream()`` which resolves against the
            # current CUDA device. For multi-GPU splits the model output (and
            # ``byte_vocab``) live on the head GPU, not cuda:0, so we must
            # match the current device or kernels launch on the wrong stream
            # and read foreign-device pointers -> cudaErrorIllegalAddress.
            torch.cuda.set_device(byte_vocab._device)

        # v8 dual-channel attachment (plan §4, §8).  When the run is configured
        # with a specials map, build the event vocab + resolved specials and
        # bind them to the byte vocab so subsequent marginalise calls produce
        # the renormalised byte channel and the event channel side-by-side.
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

                batch_tensor = torch.from_numpy(batch_tokens_np)
                batch_logits = model.forward(batch_tensor, {}).contiguous()

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

                    # v8 event payload.  Filling in row indices on the
                    # template here means the manifest written to HDF5 already
                    # carries the correct row pointers.
                    if event_dist_gpu is not None and event_dist_gpu.numel() > 0 and event_template:
                        # Some anchor positions may have been clipped by
                        # marginalise_dual; rebuild the manifest in that case
                        # by aligning template entries with positions that
                        # actually fell inside [0, length).
                        kept = []
                        kept_template = []
                        for tpl, pos in zip(event_template, anchor_positions):
                            if 0 <= int(pos) < length:
                                kept_template.append(dict(tpl))
                                kept.append(int(pos))
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
                import traceback
                print(f"  WARNING: Inference worker error: {e}")
                traceback.print_exc()
                break
    finally:
        try:
            del model
        except NameError:
            pass
        torch.cuda.empty_cache()
        result_queue.put(None)
