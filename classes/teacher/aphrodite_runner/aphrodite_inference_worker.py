import torch
import numpy as np
from multiprocessing import shared_memory
from numpy import ndarray
from aphrodite import LLM, SamplingParams
from tqdm import tqdm
    

def _inference_worker(pbar: tqdm, model_path, model_name, reserve_vram_gb, crop_to_size, context_len, topK):

    def _load_model() -> LLM:
        pbar.set_postfix_str(f"Loading {model_name}...")

        # Initialize the LLM with the specified model
        llm = LLM(model=model_path, tensor_parallel_size=1, enforce_eager=True, gpu_memory_utilization=0.8)

        return llm

    def _unload_model(llm: LLM):
        if llm is None:
            print(f"{model_name} is already unloaded.")
            return

        del llm
        torch.cuda.empty_cache()

    def _inference(llm: LLM, batch_tokenized_np: ndarray, batch_distributions) -> tuple:
        batch_tokenized = batch_tokenized_np.tolist()

        # Extract logits from the outputs
        batch_logp_list = []
        for output in outputs:
            logits = output.logits
            logp = torch.log_softmax(torch.tensor(logits)[:, :crop_to_size], dim=-1)
            batch_logp_list.append(logp)

        batch_logp = torch.stack(batch_logp_list)

        indices = None
        batch_logp, indices = torch.topk(batch_logp, topK, dim=-1)
        indices = indices.numpy()

        batch_logp_data = batch_logp.numpy()

        shd_mem = shared_memory.SharedMemory(create=True, size=batch_logp_data.nbytes)
        shared_batch_logp = np.ndarray(batch_logp_data.shape, dtype=batch_logp_data.dtype, buffer=shd_mem.buf)
        np.copyto(shared_batch_logp, batch_logp_data)
        return shd_mem.name, batch_logp_data.shape, batch_logp_data.dtype, shd_mem, indices

    llm = _load_model()
    
        
    sampling_params = SamplingParams(max_tokens=1, prompt_logprobs=topK, logprobs=topK, skip_special_tokens=False)

        
    outputs = llm.generate(batch_tokenized, sampling_params=sampling_params, tensor_parallel_size=2)

    
