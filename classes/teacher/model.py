from classes.teacher.batch_creator_worker import _batch_creator_worker
from classes.teacher.inference_worker import _inference_worker
from classes.teacher.result_processor_worker import _result_processor_worker
from classes.data_classes import ConvoTokenized
from classes.base_model import BaseModel
from tqdm import tqdm
import multiprocessing
import torch
import os
import gc


if not torch.cuda.is_initialized():
    torch.cuda.init()

os.environ['CUDA_LAUNCH_BLOCKING'] = '1'


class TeacherModel(BaseModel):
    def __init__(self, model_path: str, max_queue_size=3):
        super().__init__(model_path)
        self.distr_device: str = ""
        self.encourage_eos: bool = False
        self.stop_id: int = 0
        self.student_eos_id: int = 0
        self.reserve_vram = []
        self.max_queue_size = max_queue_size

    def process_chunk(self, reserve_vram_gb: list[float] = [], ids_to_collect: list = [], full_collect: bool = False, data_manager = None, validation: bool = False):
        self._sort_datasets_by_len()

        dataset_chunk = (self.validation_dataset if validation else self.dataset)

        if not validation and not full_collect:
            dataset_chunk = [convo for convo in dataset_chunk if convo.origin_convo_id in ids_to_collect]

        with multiprocessing.Manager() as manager:
            done_chunk = manager.Event()
            made_distributions = manager.Event()
            inference_queue = manager.Queue(self.max_queue_size)
            result_queue = manager.Queue(self.max_queue_size)
            disk_queue = manager.Queue(self.max_queue_size)
            pbar_queue = manager.Queue(self.max_queue_size)

            self.progress_bar = tqdm(total=len(dataset_chunk), desc="Convos", smoothing=0.06, leave=False)
            self.data_manager = data_manager
            self.reserve_vram = reserve_vram_gb

            workers = self._start_workers(done_chunk, made_distributions, inference_queue, result_queue, disk_queue, pbar_queue, dataset_chunk)

            while not done_chunk.is_set():
                self._manage_queues(disk_queue, pbar_queue)

            self.data_manager.done_everything.wait()
            self._stop_workers(*workers)
            self.progress_bar.close()

    def _sort_datasets_by_len(self):
        if not self.dataset_sorted:
            self.dataset.sort(key=lambda convo: convo.length, reverse=True)
            self.dataset_sorted = True

        if not self.validation_dataset_sorted:
            self.validation_dataset.sort(key=lambda convo: convo.length, reverse=True)
            self.validation_dataset_sorted = True

    def _manage_queues(self, disk_queue, pbar_queue):
        while not disk_queue.empty():
            batch_content_distributions = disk_queue.get()
            disk_queue.task_done()
            self.data_manager.write_batch(batch_content_distributions)

        while not pbar_queue.empty():
            action, value = pbar_queue.get()
            pbar_queue.task_done()
            self._pbar_actions(action, value)

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
