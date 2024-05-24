from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
from multiprocessing import shared_memory
from numpy import ndarray
import torch.nn.functional as F
import numpy as np
import torch
import math


def _inference_worker(inference_queue, result_queue, made_distributions, done_chunk, model_path, model_name, reserve_vram,
                        device, temperature, crop_to_size, pbar_queue, context_len, batch_size, max_queue_size, seq_chunk_len, enable_topK, topK):
        
    def _load_model(reserve_vram_gb: list[float] = []) -> tuple[ExLlamaV2, ExLlamaV2Cache]:
        pbar_queue.put(("str", f"Loading {model_name}..."))

        num_gpus = torch.cuda.device_count()
        if num_gpus == 0:
            raise Exception("No CUDA-capable GPUs found.")

        reserve_vram_kb = [int(128 * 1024**2)]*num_gpus # default 128MB/GPU

        for i, reserve_gb in enumerate(reserve_vram_gb):
            if reserve_gb > 0 and i < num_gpus:
                reserve_vram_kb[i] = int(reserve_gb * 1024**3)

        config = ExLlamaV2Config()
        config.model_dir = model_path
        config.prepare()
        config.max_seq_len = context_len
        config.max_batch_size = batch_size
        config.max_input_len = seq_chunk_len
        config.max_attention_size = context_len ** 2 

        model = ExLlamaV2(config)

        cache = ExLlamaV2Cache(model, batch_size, context_len, lazy=True)

        model.load_autosplit(cache, reserve_vram=reserve_vram_kb)
        return model, cache

    def _unload_model(model: ExLlamaV2, cache: ExLlamaV2Cache):
        if model is None:
            print(f"{model_name} is already unloaded, genius.")
            return

        model.unload()

        del model
        del cache
        torch.cuda.empty_cache()

    def _inference(model: ExLlamaV2, cache: ExLlamaV2Cache, batch_tokenized: torch.Tensor) -> ndarray:
        #assert model is not None, f"{model_name}: Model has not been loaded yet."

        with torch.no_grad():
            cache.current_seq_len = 0
            
            batch_logp: torch.Tensor = F.log_softmax(model.forward(batch_tokenized, cache=cache).contiguous()[:, :, :crop_to_size].float()/temperature, dim=-1)
            indices = None

            if enable_topK:
                batch_logp, indices = torch.topk(batch_logp, topK, dim=-1)
                indices: ndarray = indices.to('cpu').numpy()

            batch_logp_data: ndarray = batch_logp.to('cpu').numpy()
                
            shd_mem = shared_memory.SharedMemory(create=True, size=batch_logp_data.nbytes)
            shared_batch_logp = ndarray(batch_logp_data.shape, dtype=batch_logp_data.dtype, buffer=shd_mem.buf)
            np.copyto(shared_batch_logp, batch_logp_data)
            return shd_mem.name, batch_logp_data.shape, batch_logp_data.dtype, shd_mem, indices
        
    def _enqueue_batch_tensor():
        batch_tokenized_np, batch_distributions = inference_queue.get()
        inference_queue.task_done()

        if batch_tokenized_np is None:
            tokenized_batches.append((None, None))
            return
        
        if batch_tokenized_np.sum() == 0:
            raise Exception("All tokens are 0s")

        batch_tensor = torch.tensor(batch_tokenized_np, dtype=torch.long).to(device, non_blocking=True)
        tokenized_batches.append((batch_tensor, batch_distributions))
    
    torch.cuda.empty_cache()

    model, cache = _load_model(reserve_vram)

    pbar_queue.put(("str", f"Generating..."))

    tokenized_batches = []
    tokenized_batches_max = math.ceil(100 / batch_size)
    shared_list = []

    while not inference_queue.empty() and len(tokenized_batches) < tokenized_batches_max:
        _enqueue_batch_tensor()

    while True:
        batch_tensor, batch_distributions = tokenized_batches.pop(0)

        if batch_tensor is None:
            _unload_model(model, cache)
            result_queue.put((None, None, None, None, None))
            made_distributions.set()
            done_chunk.wait()
            break

        shd_mem_name, batch_logp_shape, batch_logp_dtype, shd_mem, indices_np = _inference(model, cache, batch_tensor)
        shared_list.append(shd_mem)
        if len(shared_list) > max_queue_size + 20:
            shared_list.pop(0)
        result_queue.put((shd_mem_name, batch_logp_shape, batch_logp_dtype, batch_distributions, indices_np))
        made_distributions.set()

        if not inference_queue.empty():
            _enqueue_batch_tensor()
