from classes.data_classes import Distribution
from multiprocessing import shared_memory
from multiprocessing import get_context
import multiprocessing
import numpy as np
import time
import h5py
import os


class H5DataManager:
    def __init__(self, dataset_path, device, max_queue_size=5):
        self.file_path = os.path.join(dataset_path, "distributions.hdf5")
        self.device = device
        self.queue = multiprocessing.Queue(max_queue_size)
        self.result_queue = multiprocessing.Queue(max_queue_size)
        self.max_queue_size = max_queue_size
        self.done_everything = multiprocessing.Event()
        self.done_everything.set()
        self.got_task = multiprocessing.Event()
        self.loading_process = get_context("spawn").Process(target=self._process_thread)
        self.shared_batches = []

        self.loading_process.start()

    def _process_thread(self):
        with h5py.File(self.file_path, 'a') as hdf_file:

            if not self.queue.empty():
                self.got_task.set()

            while True:
                self.got_task.wait()
                self.got_task.clear()
                self.done_everything.clear()
                task = None

                while not self.queue.empty():
                    task, data = self.queue.get()

                    match task:
                        case 'get_batch':
                            self.result_queue.put(self._make_outgoing_batch(hdf_file, data))
                        case 'get_batches':
                            for batch_ids in data:
                                self.result_queue.put(self._make_outgoing_batch(hdf_file, batch_ids))
                            self.got_task.wait()
                        case 'read_only_mode':
                            while not self.got_task.is_set():
                                for batch_ids in data:

                                    while self.queue.qsize() >= self.max_queue_size - 1:
                                        if self.got_task.is_set():
                                            break
                                        time.sleep(0.1)

                                    if self.got_task.is_set():
                                        break

                                    self.result_queue.put(self._make_outgoing_batch(hdf_file, batch_ids))
                                    
                        case 'put_batch':
                            self._process_distributions(hdf_file, data)
                        case 'update_batch':
                            self._update_data(hdf_file, data)
                        case 'get_available_ids':
                            self.result_queue.put([int(dataset_name.split('_')[1]) for dataset_name in hdf_file])
                        case 'clear_dataset':
                            self._clear_dataset(hdf_file)
                        case 'exit':
                            break
                            
                if task == 'exit':
                    break
                        
                self.done_everything.set()
                
    def _process_distributions(self, hdf_file, batch: list[Distribution]):
        for distribution in batch:
            distr_shd_mem = shared_memory.SharedMemory(name=distribution.shd_mem_name)
            distribution.distribution = np.ndarray(distribution.distr_shape, dtype=distribution.distr_dtype, buffer=distr_shd_mem.buf)

            id = distribution.origin_convo_id
            if f"{id}" in hdf_file:
                merged_data = self._merge_data(hdf_file, distribution, id)
                self._save_data(hdf_file, merged_data, id)
            else:
                self._save_data(hdf_file, distribution.distribution, id, distribution.indices)

            distr_shd_mem.close()
            distr_shd_mem.unlink()

    def _merge_data(self, hdf_file, new_distribution: Distribution, id):
        assert isinstance(new_distribution.distribution, np.ndarray), "Distribution should be a numpy array."
        
        ppl = new_distribution.ppl
        assert ppl > 0, "PPL should be greater than 0."

        disk_data = np.exp(self._load_id(hdf_file, id))
        new_data = np.exp(new_distribution.distribution) / np.log(ppl)

        raw_diff = disk_data.shape[0] - new_data.shape[0]

        if raw_diff < 0:
            disk_data = np.pad(disk_data, ((0, -raw_diff), (0, 0)))
        elif raw_diff > 0:
            new_data = np.pad(new_data, ((0, raw_diff), (0, 0)))

        merged_data = disk_data + new_data

        return np.log(merged_data)
    
    def _update_data(self, hdf_file, batch: list[Distribution]):
        for distribution in batch:
            distr_shd_mem = shared_memory.SharedMemory(name=distribution.shd_mem_name)
            distribution.distribution = np.ndarray(distribution.distr_shape, dtype=distribution.distr_dtype, buffer=distr_shd_mem.buf)

            self._save_data(hdf_file, distribution.distribution, distribution.origin_convo_id, distribution.indices)

            distr_shd_mem.close()
            distr_shd_mem.unlink()

    def _save_data(self, hdf_file: h5py.File, data: np.ndarray, convo_id: int, indices):
        dataset_name = f'convo_{convo_id}'
        if dataset_name in hdf_file:
            del hdf_file[dataset_name]
        hdf_file.create_dataset(dataset_name, data=data, dtype=np.float16)

        if indices is not None:
            indices_name = f'indices_{convo_id}'
            if indices_name in hdf_file:
                del hdf_file[indices_name]
            hdf_file.create_dataset(indices_name, data=indices, dtype=np.int32)
    
    def _load_id(self, hdf_file, convo_id: int) -> np.ndarray:
        data = np.array(hdf_file[f'convo_{convo_id}'], dtype=np.float32) if f'convo_{convo_id}' in hdf_file else None
        indices = np.array(hdf_file[f'indices_{convo_id}'], dtype=np.int64) if f'indices_{convo_id}' in hdf_file else None
        if data is None:
            raise ValueError(f"Convo ID {convo_id} not found in dataset.")
        return data, indices
    
    def _make_outgoing_batch(self, hdf_file, batch_ids: list[int]) -> tuple[str, tuple[int, int], np.dtype]:
        batch = []
        batch_indices = []
        for convo_id in batch_ids:
            data, indices = self._load_id(hdf_file, convo_id)
            batch.append(data)
            batch_indices.append(indices)

        max_len = max([len(distr) for distr in batch])
        padded_data = [(np.pad(distr, ((0, max_len - len(distr)), (0, 0))), (len(distr))) for distr in batch]
        padded_batch = np.array([distr[0] for distr in padded_data])
        batch_padding = [distr[1] for distr in padded_data]

        shared_batch_memory = shared_memory.SharedMemory(create=True, size=padded_batch.nbytes)
        shared_batch = np.ndarray(padded_batch.shape, dtype=padded_batch.dtype, buffer=shared_batch_memory.buf)
        np.copyto(shared_batch, padded_batch)
        self.shared_batches.append(shared_batch_memory)

        if len(self.shared_batches) >= self.max_queue_size + 10:
            self.shared_batches = self.shared_batches[1:]

        return (shared_batch_memory.name, shared_batch.shape, shared_batch.dtype, batch_indices, batch_padding)

    def _clear_dataset(self, hdf_file: h5py.File):
        for dataset_name in hdf_file:
            del hdf_file[dataset_name]
        self.shared_batches = []
        self.result_queue.put(True)
    
    def enqueue_get_batches(self, batches: list[list[int]]):
        self.queue.put(('get_batches', batches))
        self.got_task.set()

    def read_only_mode(self, batches: list[list[int]]):
        self.queue.put(('read_only_mode', batches))
        self.got_task.set()
    
    def read_next_batch(self):
        return self.result_queue.get()

    def write_batch(self, batch: list[Distribution]):
        self.queue.put(('put_batch', batch))
        self.got_task.set()

    def update_batch(self, batch: list[Distribution]):
        self.queue.put(('update_batch', batch))
        self.got_task.set()

    def get_dataset_ids(self) -> list[int]:
        self.done_everything.wait()
        self.queue.put(('get_available_ids', None))
        self.got_task.set()
        return self.result_queue.get()

    def purge_dataset(self):
        self.queue.put(('clear_dataset', None))
        self.got_task.set()
        return self.result_queue.get()

    def close(self):
        self.queue.put(('exit', None))
        self.got_task.set()
        self.loading_process.join()
