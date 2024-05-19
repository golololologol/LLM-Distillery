from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
from classes.data_classes import ConvoTokenized, Distribution
from multiprocessing import shared_memory
from classes.base_model import BaseModel
from transformers import AutoTokenizer
from numpy import ndarray
from tqdm import tqdm
import torch.nn.functional as F
import multiprocessing
import numpy as np
import torch
if not torch.cuda.is_initialized():
    torch.cuda.init()
import math
import json
import os
import gc

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'

def _batch_creator_worker(inference_queue, batch_size, dataset_chunk: list[ConvoTokenized]):
    def _prepare_batch_for_inference(convo_id, batch_size, dataset_chunk: list[ConvoTokenized]) -> tuple[ndarray, list[Distribution], int]:
        batch_tokenized: list[ndarray] = []
        batch_distributions: list[Distribution] = []

        batch_convos = dataset_chunk[convo_id:convo_id+batch_size]

        for convo in batch_convos:
            batch_distributions.append(Distribution(convo.origin_convo_id, convo.length, convo.cropped_end, convo.content_ranges, convo.tokenized))
            batch_tokenized.append(convo.tokenized)

        # Crop to the convos to one that has the least non-padding tokens
        max_non_padded_len = max(convo.length for convo in batch_convos)
        batch_tokenized = [convo_tokenized[:max_non_padded_len] for convo_tokenized in batch_tokenized]

        batch_tokenized_np = np.array(batch_tokenized)
        batch_size = len(batch_distributions)

        return batch_tokenized_np, batch_distributions, batch_size
        
    convo_id = 0
    num_batches = math.ceil(len(dataset_chunk) / batch_size)

    for i in range(num_batches):
        batch_tokenized, batch_distributions, batch_size = _prepare_batch_for_inference(convo_id, batch_size, dataset_chunk)
        inference_queue.put((batch_tokenized, batch_distributions))
        convo_id += batch_size

    inference_queue.put((None, None))


def _inference_worker(inference_queue, result_queue, made_distributions, done_chunk, model_path, model_name, reserve_vram,
                        device, temperature, crop_to_size, pbar_queue, context_len, batch_size, max_queue_size, seq_chunk_len, enable_topK, topK):
        
    def _load_model(reserve_vram_gb: list[float] = []):
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


def _result_processor_worker(result_queue, made_distributions, done_chunk_writes, disk_queue, encourage_eos, student_eos_id, pbar_queue: tqdm, max_queue_size: int, batch_size: int):
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

            shd_mem = shared_memory.SharedMemory(create=True, size=distribution.distribution.nbytes)
            distribution.shd_mem_name = shd_mem.name
            distribution.distr_shape = distribution.distribution.shape
            distribution.distr_dtype = distribution.distribution.dtype
            shd_distr = np.ndarray(distribution.distr_shape, dtype=distribution.distr_dtype, buffer=shd_mem.buf)
            np.copyto(shd_distr, distribution.distribution)
            shared_list.append(shd_mem)

            if len(shared_list) > max_queue_size + 20:
                shared_list.pop(0)

            del distribution.distribution

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

    while True:
        made_distributions.wait()
        while not result_queue.empty():
            shd_mem_name, batch_logp_shape, batch_logp_dtype, batch_distributions, indices_np = result_queue.get()
            result_queue.task_done()

            if batch_distributions is None:
                exit_flag = True
                done_chunk_writes.set()
                break

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


class TeacherModel(BaseModel):
    def __init__(self, model_path: str, max_queue_size=3):
        super().__init__(model_path)
        self.distr_device: str = ""
        self.encourage_eos: bool = False
        self.stop_id: int = 0
        self.student_eos_id: int = 0
        self.reserve_vram = []
        self.max_queue_size = max_queue_size

    def process_chunk(self, reserve_vram_gb: list[float] = [], next_stop_id: int = 1, data_manager = None, full_collect: bool = False, validation: bool = False):
        if full_collect:
            num_to_process = self.validation_dataset_len if validation else self.dataset_len
        else:
            num_to_process = next_stop_id - self.stop_id

        dataset_chunk = (self.validation_dataset if validation else self.dataset)[self.stop_id:self.stop_id+num_to_process]

        with multiprocessing.Manager() as manager:
            done_chunk = manager.Event()
            made_distributions = manager.Event()
            inference_queue = manager.Queue(self.max_queue_size)
            result_queue = manager.Queue(self.max_queue_size)
            disk_queue = manager.Queue(self.max_queue_size)
            pbar_queue = manager.Queue(self.max_queue_size)

            self.progress_bar = tqdm(total=num_to_process, desc="Convos", smoothing=0.06, leave=False)
            self.data_manager = data_manager
            self.reserve_vram = reserve_vram_gb

            workers = self._start_workers(done_chunk, made_distributions, inference_queue, result_queue, disk_queue, pbar_queue, dataset_chunk)

            while not done_chunk.is_set():
                self._manage_queues(disk_queue, pbar_queue)

            self.data_manager.done_everything.wait()
            self._stop_workers(*workers)
            self.progress_bar.close()

            if not validation:
                self.stop_id = next_stop_id
    
    def new_epoch(self):
        self.stop_id = 0

    def _manage_queues(self, disk_queue, pbar_queue):
        while not disk_queue.empty():
            batch_content_distributions = disk_queue.get()
            disk_queue.task_done()
            self.data_manager.write_batch(batch_content_distributions)

        while not pbar_queue.empty():
            action, value = pbar_queue.get()
            pbar_queue.task_done()
            self._pbar_actions(action, value)

    def sort_dataset_by_len(self):
        self.validation_dataset.sort(key=lambda convo: convo.length, reverse=True)
        self.validation_dataset_sorted = True
        validation_sorting_map = {convo.origin_convo_id: i for i, convo in enumerate(self.validation_dataset)}

        self.dataset.sort(key=lambda convo: convo.length, reverse=True)
        self.dataset_sorted = True
        sorting_map = {convo.origin_convo_id: i for i, convo in enumerate(self.dataset)}
        return sorting_map, validation_sorting_map
    
    def sort_dataset_by_map(self, sorting_map: dict[int, int], validation: bool = False):
        if validation:
            self.validation_dataset.sort(key=lambda convo: sorting_map[convo.origin_convo_id])
            self.validation_dataset_sorted = True
        else:
            self.dataset.sort(key=lambda convo: sorting_map[convo.origin_convo_id])
            self.dataset_sorted = True

    def _pbar_actions(self, action: str, value: int):
        if action == "increment":
            self.progress_bar.update(value)
        elif action == "str":
            self.progress_bar.set_postfix_str(value)
    
    def _unload_full(self):
        self.progress_bar.set_postfix_str(f"Fully unloading {self.model_name}...")
        self._stop_workers()
        self.dataset = []
        self.dataset_len = 0
        self.validation_dataset = []
        self.stop_id = 0
        self.dataset_sorted = False
        self.validation_dataset_sorted = False
        gc.collect()
    
    def _start_workers(self, done_chunk, made_distributions, inference_queue, result_queue, disk_queue, pbar_queue, dataset_chunk: list[ConvoTokenized]):
        self.progress_bar.set_postfix_str(f"Starting workers for {self.model_name}...")

        batch_creator = multiprocessing.Process(target=_batch_creator_worker, args=(inference_queue, self.batch_size, dataset_chunk))
        inference_worker = multiprocessing.Process(target=_inference_worker, args=(inference_queue, result_queue, made_distributions, done_chunk, self.model_path, self.model_name, self.reserve_vram, self.device, 
                                                                                   self.temperature, self.crop_to_size, pbar_queue, self.context_len, self.batch_size, self.max_queue_size, self.seq_chunk_len, self.enable_topK, self.topK))
        result_processor = multiprocessing.Process(target=_result_processor_worker, args=(result_queue, made_distributions, done_chunk, disk_queue, self.encourage_eos, self.student_eos_id, pbar_queue, self.max_queue_size, self.batch_size))

        batch_creator.start()
        inference_worker.start()
        result_processor.start()

        return batch_creator, inference_worker, result_processor
    
    def _stop_workers(self, batch_creator: multiprocessing.Process, inference_worker: multiprocessing.Process, result_processor: multiprocessing.Process):
        self.progress_bar.set_postfix_str(f"Stopping workers for {self.model_name}...")
        batch_creator.join()
        inference_worker.join()
        result_processor.terminate()
