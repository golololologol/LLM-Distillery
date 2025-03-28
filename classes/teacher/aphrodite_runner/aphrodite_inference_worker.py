import signal
import torch
import numpy as np
from multiprocessing import shared_memory
from numpy import ndarray
from aphrodite import LLM, SamplingParams


    # bogus code, WIP
    

def _inference_worker(inference_queue, result_queue, made_distributions, model_loaded, start_inference, done_chunk, model_path, model_name, reserve_vram_gb,
                      gpus_mem_used, crop_to_size, pbar_queue, context_len, batch_size, max_queue_size, seq_chunk_len, enable_topK, topK):
    def _signal_handler(signum, frame):
        while not inference_queue.empty():
            inference_queue.get()
        exit(0)
        
        
        
    
    signal.signal(signal.SIGINT, _signal_handler)
    signal.signal(signal.SIGTERM, _signal_handler)
    signal.signal(signal.SIGABRT, _signal_handler)

    def _load_model() -> LLM:
        pbar_queue.put(("str", f"Loading {model_name}..."))

        # Initialize the LLM with the specified model
        llm = LLM(model=model_name)
        model_loaded.set()

        return llm

    def _unload_model(llm: LLM):
        if llm is None:
            print(f"{model_name} is already unloaded.")
            return

        del llm
        torch.cuda.empty_cache()

    def _inference(llm: LLM, batch_tokenized_np: ndarray, batch_distributions) -> tuple:
        batch_tokenized = batch_tokenized_np.tolist()

        # Define sampling parameters
        sampling_params = SamplingParams(max_tokens=0, top_k=topK)  # Set max_tokens to 0 to prevent generation

        # Perform the forward pass to get logits
        outputs = llm.generate(batch_tokenized, sampling_params=sampling_params, tensor_parallel_size=2)

        # Extract logits from the outputs
        batch_logp_list = []
        for output in outputs:
            logits = output.logits
            logp = torch.log_softmax(torch.tensor(logits)[:, :crop_to_size], dim=-1)
            batch_logp_list.append(logp)

        batch_logp = torch.stack(batch_logp_list)

        indices = None
        if enable_topK:
            batch_logp, indices = torch.topk(batch_logp, topK, dim=-1)
            indices = indices.numpy()

        batch_logp_data = batch_logp.numpy()

        shd_mem = shared_memory.SharedMemory(create=True, size=batch_logp_data.nbytes)
        shared_batch_logp = np.ndarray(batch_logp_data.shape, dtype=batch_logp_data.dtype, buffer=shd_mem.buf)
        np.copyto(shared_batch_logp, batch_logp_data)
        return shd_mem.name, batch_logp_data.shape, batch_logp_data.dtype, shd_mem, indices

    llm = _load_model()

    pbar_queue.put(("str", "Generating..."))

    shared_list = []

    while True:
        if done_chunk.is_set():
            _unload_model(llm)
            break

        if inference_queue.empty():
            continue

        batch_tokenized_np, batch_distributions = inference_queue.get()

        if batch_tokenized_np is None:
            _unload_model(llm)
            result_queue.put((None, None, None, None, None))
            made_distributions.set()
            done_chunk.wait()
            break

        shd_mem_name, batch_logp_shape, batch_logp_dtype, shd_mem, indices_np = _inference(llm, batch_tokenized_np, batch_distributions)

        shared_list.append(shd_mem)
        if len(shared_list) > max_queue_size + 20:
            shared_list.pop(0)

        result_queue.put((shd_mem_name, batch_logp_shape, batch_logp_dtype, batch_distributions, indices_np))
        made_distributions.set()
        start_inference.wait()
