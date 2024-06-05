from classes.data_classes import Distribution
from multiprocessing import shared_memory
from numpy import ndarray
from tqdm import tqdm
import numpy as np


def _result_processor_worker(result_queue, made_distributions, done_chunk_writes, disk_queue, encourage_eos, student_eos_id, pbar_queue: tqdm, max_queue_size: int, batch_size: int, num_inference_workers: int):
    def _get_content_indices_np(content_ranges) -> ndarray:
        content_indices = []
        for start, end in content_ranges:
            content_indices.append(np.arange(start, end))
        return np.concatenate(content_indices)
        
    def _get_batch_content_logprobs(batch_distributions: list[Distribution]) -> list[Distribution]:
        batch_content_distributions = []
        batch_distributions_len = len(batch_distributions)

        for distribution in batch_distributions:
            content_indices = _get_content_indices_np(distribution.content_ranges)
            distribution.distribution = distribution.distribution[content_indices]
            distribution.indices = distribution.indices[:distribution.length] if distribution.indices is not None else None
            distribution.indices = distribution.indices[content_indices] if distribution.indices is not None else None

            shd_mem = distribution.to_shd_mem()
            shared_list.append(shd_mem)

            if len(shared_list) > max_queue_size + 40:
                shared_list.pop(0)

            batch_content_distributions.append(distribution)

        return batch_content_distributions, batch_distributions_len

    def _process_inference_results(batch_logprobs_np: ndarray, batch_distributions: list[Distribution], batch_indices_np: ndarray) -> int:
        for i, distribution in enumerate(batch_distributions):
            convo_logprobs = batch_logprobs_np[i]
            if batch_indices_np is not None:
                convo_indices = batch_indices_np[i]
            
            if encourage_eos:
                content_end_ids = [end-1 for start, end in distribution.content_ranges]

                if distribution.cropped_end:
                    content_end_ids = content_end_ids[:-1]

                for end in content_end_ids:
                    eos_pos_logprob = np.log((np.max(np.exp(convo_logprobs[end])) * 1.1))
                    if batch_indices_np is not None:

                        if student_eos_id not in convo_indices[end]:
                            convo_logprobs[end, -1:] = eos_pos_logprob
                            convo_indices[end][-1] = student_eos_id
                        else:
                            position = np.where(convo_indices[end] == student_eos_id)[0][0]
                            convo_logprobs[end, position] = eos_pos_logprob

                    else:
                        convo_logprobs[end, student_eos_id] = eos_pos_logprob
                        convo_logprobs[end] = np.log((np.exp(convo_logprobs[end]) / np.sum(np.exp(convo_logprobs[end]))))

            distribution.distribution = convo_logprobs[:distribution.length]
            distribution.indices = convo_indices
        batch_content_distributions, batch_distributions_len = _get_batch_content_logprobs(batch_distributions)
        disk_queue.put(batch_content_distributions)

        return batch_distributions_len
    
    
    exit_flag = False
    shared_list = []
    num_inference_exits = 0

    while True:
        made_distributions.wait()
        while not result_queue.empty():
            shd_mem_name, batch_logp_shape, batch_logp_dtype, batch_distributions, indices_np = result_queue.get()
            result_queue.task_done()

            if batch_distributions is None:
                num_inference_exits += 1
                if num_inference_exits == num_inference_workers:
                    exit_flag = True
                    done_chunk_writes.set()
                    break
                else:
                    continue

            logp_shd_mem = shared_memory.SharedMemory(name=shd_mem_name)
            batch_logprobs_np = np.ndarray(batch_logp_shape, dtype=batch_logp_dtype, buffer=logp_shd_mem.buf)

            batch_distributions_len = _process_inference_results(batch_logprobs_np, batch_distributions, indices_np)
            logp_shd_mem.close()
            logp_shd_mem.unlink()

            pbar_queue.put(("increment", batch_distributions_len))
        
        made_distributions.clear()

        if exit_flag:
            # block until terminated
            made_distributions.wait()