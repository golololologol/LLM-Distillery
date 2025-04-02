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
        self.progress_bar: tqdm = None

    def _sort_datasets_by_len(self):
        if not self.dataset_sorted:
            self.dataset.sort(key=lambda convo: convo.length, reverse=True)
            self.dataset_sorted = True

        if not self.validation_dataset_sorted:
            self.validation_dataset.sort(key=lambda convo: convo.length, reverse=True)
            self.validation_dataset_sorted = True


            
    def process_chunk(self, reserve_vram_gb: list[float] = [], num_inference_workers: int = 1, ids_to_collect: list = [], data_manager: H5DataManager = None, validation: bool = False):
        self._sort_datasets_by_len()

        dataset_chunk = (self.validation_dataset if validation else self.dataset)
        dataset_chunk = [convo for convo in dataset_chunk if convo.origin_convo_id in ids_to_collect]



    def close(self):
        if self.progress_bar is not None:
            self.progress_bar.set_postfix_str(f"Fully unloading {self.model_name}...")
        self._stop_workers()
        self.dataset = []
        self.dataset_len = 0
        self.validation_dataset = []
        self.stop_id = 0
        self.dataset_sorted = False
        self.validation_dataset_sorted = False
        
        if self.progress_bar is not None:
            self.progress_bar.close()
        torch.cuda.empty_cache()
        gc.collect()
