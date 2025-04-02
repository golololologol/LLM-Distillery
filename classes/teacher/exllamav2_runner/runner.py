from classes.teacher.exllamav2_runner.batch_creator_worker import _batch_creator_worker
from classes.teacher.exllamav2_runner.inference_worker import _inference_worker
from classes.teacher.exllamav2_runner.result_processor_worker import _result_processor_worker
from classes.data_classes import ConvoTokenized
from classes.data_manager import H5DataManager
from classes.base_model import BaseModel
from multiprocessing import get_context
from tqdm import tqdm
import multiprocessing.spawn
import multiprocessing
import nvidia_smi
import torch
import time
import os
import gc



class InferenceRunner:
    def __init__(self, model_path: str, max_queue_size=3):
        
    def _manage_queues(self, pbar_queue):
        while not pbar_queue.empty():
            action, value = pbar_queue.get()
            match action:
                case "increment":
                    self.progress_bar.update(value)
                case "str":
                    self.progress_bar.set_postfix_str(value)
        time.sleep(0.05)

    def _start_workers(self, done_chunk, num_inference_workers, made_distributions, model_loaded, start_inference, inference_queue, result_queue, disk_queue, pbar_queue, dataset_chunk: list[ConvoTokenized]):
        self.progress_bar.set_postfix_str(f"Starting workers for {self.model_name[:20]}...")

        self.batch_creator = multiprocessing.Process(target=_batch_creator_worker, args=(inference_queue, self.batch_size, dataset_chunk, num_inference_workers))
        self.result_processor = multiprocessing.Process(target=_result_processor_worker, args=(result_queue, made_distributions, done_chunk, disk_queue, pbar_queue, self.max_queue_size, num_inference_workers))

        self.batch_creator.start()
        
        self.inference_workers: list[multiprocessing.Process] = []

        for _ in range(num_inference_workers):
            model_loaded.clear()

            gpus_mem_used = []
            for i in range(torch.cuda.device_count()):
                handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
                info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
                gpus_mem_used.append(info.used)

            worker = get_context("spawn").Process(target=_inference_worker, args=(inference_queue, result_queue, made_distributions, model_loaded, start_inference, done_chunk, self.model_path, self.model_name[:20], self.reserve_vram, gpus_mem_used, 
                                                                            self.crop_to_size, pbar_queue, self.context_len, self.batch_size, self.max_queue_size, self.seq_chunk_len, self.enable_topK, self.topK), daemon=True)

            worker.start()
            self.inference_workers.append(worker)
            model_loaded.wait()

        start_inference.set()

        self.result_processor.start()
    
    def _stop_workers(self):
        if self.progress_bar is not None:
            self.progress_bar.set_postfix_str(f"Stopping workers for {self.model_name}...")

        if self.batch_creator is not None and self.batch_creator.is_alive():
            self.batch_creator.terminate()

        for worker in self.inference_workers:
            if worker.is_alive():
                worker.terminate()

    if self.result_processor is not None and self.result_processor.is_alive():
        self.result_processor.terminate()
            
        with multiprocessing.Manager() as manager:
        done_chunk = manager.Event()
        made_distributions = manager.Event()
        model_loaded = manager.Event()
        start_inference = manager.Event()
        inference_queue = manager.Queue(self.max_queue_size)
        result_queue = manager.Queue(self.max_queue_size)
        pbar_queue = manager.Queue(self.max_queue_size)

        self.progress_bar = tqdm(total=len(dataset_chunk), desc="Convos", smoothing=0.06, leave=False)
        self.data_manager = data_manager
        self.reserve_vram = reserve_vram_gb

        self._start_workers(done_chunk, num_inference_workers, made_distributions, model_loaded, start_inference, inference_queue, result_queue, self.data_manager.queue, pbar_queue, dataset_chunk)

        while True:
            self._manage_queues(pbar_queue)
            if done_chunk.is_set() and self.data_manager.done_everything.is_set() and pbar_queue.empty():
                break

        self.data_manager.done_everything.wait()
        self._stop_workers()
        self.progress_bar.close()