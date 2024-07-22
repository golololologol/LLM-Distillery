import multiprocessing.spawn
from classes.teacher.batch_creator_worker import _batch_creator_worker
from classes.teacher.inference_worker import _inference_worker
from classes.teacher.result_processor_worker import _result_processor_worker
from classes.data_classes import ConvoTokenized
from classes.base_model import BaseModel
from multiprocessing import get_context
from tqdm import tqdm
import multiprocessing
import nvidia_smi
import torch
import time
import os
import gc


if not torch.cuda.is_initialized():
    torch.cuda.init()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class TeacherModel(BaseModel):
    def __init__(self, model_path: str, max_queue_size=3):
        super().__init__(model_path)
        self.distr_device: str = ""
        self.stop_id: int = 0
        self.reserve_vram = []
        self.max_queue_size = max_queue_size
        self.batch_creator: multiprocessing.Process = None
        self.result_processor: multiprocessing.Process = None
        self.inference_workers: list[multiprocessing.Process] = []

    def process_chunk(self, reserve_vram_gb: list[float] = [], num_inference_workers: int = 1, ids_to_collect: list = [], data_manager = None, validation: bool = False):
        self._sort_datasets_by_len()

        dataset_chunk = (self.validation_dataset if validation else self.dataset)
        dataset_chunk = [convo for convo in dataset_chunk if convo.origin_convo_id in ids_to_collect]

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
                if done_chunk.is_set() and self.data_manager.queue.empty() and pbar_queue.empty():
                    break

            self.data_manager.done_everything.wait()
            self._stop_workers()
            self.progress_bar.close()

    def _sort_datasets_by_len(self):
        if not self.dataset_sorted:
            self.dataset.sort(key=lambda convo: convo.length, reverse=True)
            self.dataset_sorted = True

        if not self.validation_dataset_sorted:
            self.validation_dataset.sort(key=lambda convo: convo.length, reverse=True)
            self.validation_dataset_sorted = True

    def _manage_queues(self, pbar_queue):
        while not pbar_queue.empty():
            action, value = pbar_queue.get()
            self._pbar_actions(action, value)
        time.sleep(0.05)

    def _pbar_actions(self, action: str, value: int):
        if action == "increment":
            self.progress_bar.update(value)
        elif action == "str":
            self.progress_bar.set_postfix_str(value)

    def _unload_full(self):
        if self.progress_bar is not None:
            self.progress_bar.set_postfix_str(f"Fully unloading {self.model_name}...")
        self._stop_workers()
        self.dataset = []
        self.dataset_len = 0
        self.validation_dataset = []
        self.stop_id = 0
        self.dataset_sorted = False
        self.validation_dataset_sorted = False
        gc.collect()
    
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

    def close(self):
        self._unload_full()
        if self.progress_bar is not None:
            self.progress_bar.close()
        torch.cuda.empty_cache()
        gc.collect()
