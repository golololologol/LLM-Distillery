from classes.data_classes import Distribution
from multiprocessing import shared_memory
from multiprocessing import get_context
import multiprocessing
import numpy as np
import traceback
import torch
import time
import h5py
import gc
import os


class H5DataManager:
    """
    DataManager class for handling the distribution data in a HDF5 file.\\
    It is asynchronous and uses multiprocessing to handle the data loading and saving.\\
    This class is designed to be used with the `Distribution` class from `classes/data_classes.py`.
    
    Note: If multiple teachers were used, logsoftmax the distributions before usage in training.
    """
    def __init__(self, dataset_path, device, max_queue_size=7, manager_name=""):
        self.file_path = os.path.join(dataset_path, "distributions.hdf5")
        self.device = device
        self.queue = multiprocessing.Queue(max_queue_size)
        self.result_queue = multiprocessing.Queue(max_queue_size)
        self.closing = multiprocessing.Event()
        self.done_everything = multiprocessing.Event()
        self.done_everything.set()
        self.max_queue_size = max_queue_size
        self.loading_process = get_context("spawn").Process(target=self._loading_process)
        self.shared_batches: list[shared_memory.SharedMemory] = []
        self.distr_key = "distributions"
        self.indices_key = "indices"
        self.manager_name = manager_name + " " if manager_name else manager_name

        self.loading_process.start()

    def _loading_process(self):
        """
        Child process method:
        1. Opens the HDF5 file.
        2. Loops for tasks from self.queue until self.closing is set.
        3. Handles each task, updates or reads from the HDF5.
        4. Closes file and cleans up queues upon getting the closing signal.
        """

        def handle_exit(signum, frame):
            self.done_everything.clear()
            hdf_file.flush()
            hdf_file.close()

            if self.queue is not None:
                while not self.queue.empty():
                    self.queue.get()
                self.queue.close()
                self.queue.join_thread()

            if self.result_queue is not None:
                while not self.result_queue.empty():
                    self.result_queue.get()
                self.result_queue.close()
                self.result_queue.join_thread()

            for shared_batch in self.shared_batches:
                shared_batch.close()
                shared_batch.unlink()
                
            del self.shared_batches
            gc.collect()
            self.done_everything.set()

        global hdf_file
        hdf_file = h5py.File(self.file_path, 'a')

        try:
            while True:
                if self.closing.is_set():
                    break

                if self.queue.empty():
                    time.sleep(0.05)
                    continue

                self.done_everything.clear()
                task, data = self.queue.get()

                match task:
                    case 'get_batch':
                        self.result_queue.put(self._make_outgoing_batch(hdf_file, data))
                    case 'get_batches':
                        self._get_batches(data)
                    case 'read_only_mode':
                        self._read_only_mode(data)
                    case 'put_batch':
                        self._process_distributions(hdf_file, data)
                    case 'update_batch':
                        self._update_data(hdf_file, data)
                    case 'clear_dataset':
                        ids_to_clear = [int(group.split('_')[1]) for group in hdf_file]
                        self._clear_dataset(hdf_file, ids_to_clear)
                    case 'clear_ids':
                        self._clear_dataset(hdf_file, data)
                    case 'rename_ids':
                        self._rename_ids(hdf_file, data)
                    case 'get_available_ids':
                        self.result_queue.put([int(group.split('_')[1]) for group in hdf_file])
                    case 'get_available_shas':
                        self.result_queue.put(self._get_shas(hdf_file))
                    case 'update_shas':
                        self._update_shas(hdf_file, data)
                    case 'get_vocab_family':
                        self.result_queue.put(self._get_vocab_family(hdf_file))
                    case 'set_vocab_family':
                        self._set_vocab_family(hdf_file, data)
                    case 'get_topk':
                        self.result_queue.put(self._get_topk(hdf_file))
                    case 'set_topk':
                        self._set_topk(hdf_file, data)
                    case 'get_dataset_attr':
                        self.result_queue.put(self._get_attr(hdf_file, data))
                    case 'set_dataset_attr':
                        self._set_attr(hdf_file, data[0], data[1])
                    case 'has_data':
                        self.result_queue.put(self._has_data(hdf_file))

                if self.queue.empty():
                    self.done_everything.set()

        except Exception as e:
            print(f"Data Manager process exception: {e}")
            traceback.print_exc()
        finally:
            handle_exit(None, None)


    def _get_attr(self, hdf_file: h5py.File, arg):
        return hdf_file.attrs.get(arg, None)
    
    def _set_attr(self, hdf_file: h5py.File, arg, value):
        hdf_file.attrs[arg] = value
    
    def _get_vocab_family(self, hdf_file: h5py.File) -> str:
        return hdf_file.attrs.get('vocab_family', None)
    
    def _set_vocab_family(self, hdf_file: h5py.File, vocab_family: str):
        hdf_file.attrs['vocab_family'] = vocab_family
    
    def _get_topk(self, hdf_file: h5py.File):
        return hdf_file.attrs.get('topk', None)
    
    def _set_topk(self, hdf_file: h5py.File, topk: int):
        hdf_file.attrs['topk'] = topk

    def _has_data(self, hdf_file: h5py.File):
        return len(hdf_file.keys()) > 0
    

    def _get_batches(self, data):
        for batch_ids in data:
            while self.result_queue.full():
                if self.closing.is_set():
                    return
                time.sleep(0.05)
            if self.closing.is_set():
                return
            self.result_queue.put(self._make_outgoing_batch(hdf_file, batch_ids))


    def _read_only_mode(self, data):
        while not self.closing.is_set():
            for batch_ids in data:
                while self.result_queue.full():
                    if self.closing.is_set():
                        return
                    time.sleep(0.05)
                if self.closing.is_set():
                    return
                self.result_queue.put(self._make_outgoing_batch(hdf_file, batch_ids))
        
 
    def _process_distributions(self, hdf_file: h5py.File, batch: list[Distribution]):
        for distribution in batch:
            shd_mem = distribution.from_shd_mem()

            id = distribution.origin_convo_id
            name = f'convo_{id}'
            if name in hdf_file:
                merged_data, merged_indices = self._merge_data(hdf_file, distribution, id)
                self._save_data(hdf_file, merged_data, id, merged_indices)
            else:
                self._save_data(hdf_file, distribution.distribution, id, distribution.indices, distribution.content_sha, distribution.sample)

            shd_mem.close()
            shd_mem.unlink()


    def _merge_data(self, hdf_file: h5py.File, new_distribution: Distribution, id) -> tuple[np.ndarray, np.ndarray]:

        def merge_full(disk_data, new_data):
            raw_diff = disk_data.shape[0] - new_data.shape[0]

            if raw_diff < 0:
                disk_data = np.pad(disk_data, ((0, -raw_diff), (0, 0)))
            elif raw_diff > 0:
                new_data = np.pad(new_data, ((0, raw_diff), (0, 0)))

            return disk_data + new_data
                
        def merge_topk(disk_data: torch.Tensor, new_data: torch.Tensor, disk_indices: torch.Tensor, new_indices: torch.Tensor):
            raw_diff = disk_data.size(0) - new_data.size(0)
            topK = disk_data.size(-1)
            
            if raw_diff < 0:
                rest_data = new_data[disk_data.size(0):]
                rest_indices = new_indices[disk_data.size(0):]
                new_data = new_data[:disk_data.size(0)]
                new_indices = new_indices[:disk_data.size(0)]
                
            elif raw_diff > 0:
                rest_data = disk_data[new_data.size(0):]
                rest_indices = disk_indices[new_data.size(0):]
                disk_data = disk_data[:new_data.size(0)]
                disk_indices = disk_indices[:new_data.size(0)]

            num_tokens = disk_data.size(0)
            merged_data = torch.zeros_like(disk_data)
            merged_indices = torch.zeros_like(disk_indices, dtype=torch.long)
    
            for i in range(num_tokens):
                all_indices = torch.cat((disk_indices[i], new_indices[i]))
                all_values = torch.cat((disk_data[i], new_data[i]))
        
                unique_indices, inv_indices = torch.unique(all_indices, return_inverse=True)
                summed_values = torch.zeros_like(unique_indices, dtype=all_values.dtype)
                summed_values.index_add_(0, inv_indices, all_values)
        
                top_values, top_indices = torch.topk(summed_values, k=topK)
                merged_data[i] = top_values
                merged_indices[i] = unique_indices[top_indices]

            if raw_diff != 0:
                merged_data = torch.cat((merged_data, rest_data))
                merged_indices = torch.cat((merged_indices, rest_indices))

            return merged_data.numpy(), merged_indices.numpy()
        

        disk_data, disk_indices = self._load_id(hdf_file, id)
        disk_data = np.exp(disk_data)
        new_data = np.exp(new_distribution.distribution)

        if disk_indices is not None:
            merged_data, merged_indices = merge_topk(torch.tensor(disk_data, dtype=torch.float32),
                                     torch.tensor(new_data, dtype=torch.float32),
                                     torch.tensor(disk_indices, dtype=torch.long),
                                     torch.tensor(new_distribution.indices, dtype=torch.long))
        else:
            merged_data = merge_full(disk_data, new_data)

        return np.log(merged_data), merged_indices
    

    def _update_data(self, hdf_file: h5py.File, batch: list[Distribution]):
        for distribution in batch:
            shd_mem = distribution.from_shd_mem()

            self._save_data(hdf_file, distribution.distribution, distribution.origin_convo_id, distribution.indices, distribution.content_sha, distribution.sample)

            shd_mem.close()
            shd_mem.unlink()
        

    def _get_shas(self, hdf_file: h5py.File) -> dict[int, str]:
        shas = {}
        for group in hdf_file:
            shas[int(group.split('_')[1])] = hdf_file[group].attrs['content_sha']
        return shas
    
    def _update_shas(self, hdf_file: h5py.File, shas: dict[int, str]):
        for id, sha in shas.items():
            group_key = f'convo_{id}'
            if group_key in hdf_file:
                hdf_file[group_key].attrs['content_sha'] = sha


    def _save_data(self, hdf_file: h5py.File, data: np.ndarray, convo_id: int, indices: np.ndarray, content_sha: str, sample: dict = {}):
        group_key = f'convo_{convo_id}'
        compression_opts = 6

        # Create or get the group
        if group_key not in hdf_file:
            group = hdf_file.create_group(group_key)
        else:
            group = hdf_file[group_key]

        if content_sha is not None:
            group.attrs['content_sha'] = content_sha

        if sample is not None:
            group.attrs['sample'] = sample

        # Create or replace the dataset for data
        if self.distr_key in group:
            del group[self.distr_key]
        group.create_dataset(self.distr_key, data=data, compression='gzip', compression_opts=compression_opts, dtype=np.float16)

        # Create or replace the dataset for indices, if provided
        if indices is not None:
            if self.indices_key in group:
                del group[self.indices_key]
            group.create_dataset(self.indices_key, data=indices, compression='gzip', compression_opts=compression_opts, dtype=np.uint32)
    

    def _load_id(self, hdf_file: h5py.File, convo_id: int) -> tuple[np.ndarray, np.ndarray]:
        group_key = f'convo_{convo_id}'
    
        if not group_key in hdf_file:
            raise ValueError(f"Convo ID {convo_id} not found in dataset.")
        
        group = hdf_file[group_key]

        distributions = np.array(group[self.distr_key], dtype=np.float32) if self.distr_key in group else None
        indices = np.array(group[self.indices_key], dtype=np.int64) if self.indices_key in group else None

        if distributions is None:
            raise ValueError(f"Convo ID {convo_id} has no distributions in dataset.")

        return distributions, indices
    

    def _make_outgoing_batch(self, hdf_file: h5py.File, batch_ids: list[int]) -> tuple[str, tuple[int, int], np.dtype, list[np.ndarray], list[int]]:
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


    def _clear_dataset(self, hdf_file: h5py.File, ids_to_clear: list[int] = []):
        for id in ids_to_clear:
            try:
                del hdf_file[f'convo_{id}']
            except:
                pass

            if f'indices_{id}' in hdf_file:
                del hdf_file[f'indices_{id}']

        self.shared_batches = []
        del hdf_file.attrs['vocab_family']
        del hdf_file.attrs['topk']


    def _rename_ids(self, hdf_file: h5py.File, ids_to_rename: dict[int, int]):
        for old_id, new_id in ids_to_rename.items():
            hdf_file.move(f'convo_{old_id}', f'convo_{new_id}_moved')

        for old_id, new_id in ids_to_rename.items():
            hdf_file.move(f'convo_{new_id}_moved', f'convo_{new_id}')
            
    
    def enqueue_get_batches(self, batches: list[list[int]]):
        """
        Enqueues a request to get padded batches from the dataset asynchronously.\n
        Each subsequent batch can be retrieved with `read_next_batch`.
        """
        self.queue.put(('get_batches', batches))

    def read_next_batch(self):
        """
        Returns the next batch of distributions from the dataset loaded asynchonously.\n
        To use this function, you must call `enqueue_get_batches` first with a list of batch IDs as List[List[int]].
        """
        return self.result_queue.get()

    def read_only_mode(self, batches: list[list[int]]):
        """
        Enables read-only mode for the datamanager.\n
        It will iterate over the given batch ids indefinitely.
        """
        self.queue.put(('read_only_mode', batches))

    def write_batch(self, batch: list[Distribution]):
        """
        Writes a batch of distributions to the dataset.\n
        Automatically merges the data if the convo_id already exists.
        """
        self.queue.put(('put_batch', batch))

    def update_batch(self, batch: list[Distribution]):
        """
        Rewrites a batch of distributions in the dataset.\n
        Does not merge the data, but replaces it.
        """
        self.queue.put(('update_batch', batch))

    def get_dataset_ids(self) -> list[int]:
        """
        Returns a list of all conversation IDs in the dataset.
        """
        self.queue.put(('get_available_ids', None))
        return self.result_queue.get()

    def purge_dataset(self, ask_confirmation=True):
        """
        Deletes all distributions from the dataset.
        """
        true_replies = ['y', 'yes', 'ye', '1', 'true', 't']

        if ask_confirmation:
            reply = input("The script is going to delete all distributions from the h5 dataset.\nAre you sure you want to proceed? (y/n):")
            if reply.lower() not in true_replies:
                raise ValueError("User cancelled operation.")
        
            reply = input("Are you REALLY sure you want to delete all distributions from the dataset? (y/n): ")
            if reply.lower() not in true_replies:
                raise ValueError("User cancelled operation.")
        
        self.queue.put(('clear_dataset', None))
        self.done_everything.wait()
    
    def delete_ids(self, ids: list[int]):
        """
        Deletes distributions with the given IDs from the dataset.
        """
        if not ids:
            return
        
        response = input(f"The script called for deletion of {len(ids)} samples from the {self.manager_name}h5 dataset.\nAre you sure you want to proceed? (y/n):")
        if response.lower() not in ['y', 'yes', 'ye', '1', 'true', 't']:
            raise ValueError("User cancelled operation.")
        
        self.queue.put(('clear_ids', ids))
        self.done_everything.wait()

    def rename_ids(self, ids_to_rename: dict[int, int]):
        """
        Renames the given IDs in the dataset.
        """
        if not ids_to_rename:
            return
        
        response = input(f"The script called for renaming of {len(ids_to_rename)} samples in the {self.manager_name}h5 dataset.\nThis means that the dataset's samples will be moved to new IDs to be in sync with your current text dataset.\nAre you sure you want to proceed? (y/n):")
        if response.lower() not in ['y', 'yes', 'ye', '1', 'true', 't']:
            raise ValueError("User cancelled operation.")
        
        self.queue.put(('rename_ids', ids_to_rename))
        self.done_everything.wait()

    def sync(self, ids_to_delete, ids_to_rename):
        """
        Deletes and renames the given IDs in the dataset.\n
        Used for syncing the h5 file with the current dataset.\n
        This function is synchronous and will block until the operation is complete.
        """
        self.done_everything.wait()
        self.delete_ids(ids_to_delete)

        self.done_everything.wait()
        self.rename_ids(ids_to_rename)
        
        self.done_everything.wait()

    def get_available_shas(self) -> dict[str, str]:
        """
        Returns a dictionary of all conversation IDs and their content SHA hashes.
        """
        self.done_everything.wait()
        self.queue.put(('get_available_shas', None))
        return self.result_queue.get()
    
    def update_shas(self, shas: dict[int, str]):
        """
        Updates the content SHA hashes of the given conversation IDs in the dataset.
        """
        self.queue.put(('update_shas', shas))
        self.done_everything.wait()

    def set_dataset_attr(self, attr: str, value):
        """
        Sets an attribute for the dataset.
        """
        self.queue.put(('set_dataset_attr', (attr, value)))

    def get_dataset_attr(self, attr: str):
        """
        Gets an attribute from the dataset.
        """
        self.queue.put(('get_dataset_attr', attr))
        return self.result_queue.get()
    
    def get_vocab_family(self) -> str:
        """
        Returns the vocabulary family of the dataset.
        """
        self.done_everything.wait()
        self.queue.put(('get_vocab_family', None))
        return self.result_queue.get()
    
    def set_vocab_family(self, vocab_family: str):
        """
        Sets the vocabulary family of the dataset.
        """
        self.done_everything.wait()
        self.queue.put(('set_vocab_family', vocab_family))
        self.done_everything.wait()

    def get_topk(self) -> int:
        """
        Returns the TopK value with which the dataset was collected.
        """
        self.done_everything.wait()
        self.queue.put(('get_topk', None))
        return self.result_queue.get()
    
    def set_topk(self, topk: int):
        """
        Sets the TopK value of the dataset.
        """
        self.done_everything.wait()
        self.queue.put(('set_topk', topk))
        self.done_everything.wait()
    
    def has_data(self) -> bool:
        """
        Returns whether the dataset has any distributions saved.
        """
        self.queue.put(('has_data', None))
        self.done_everything.wait()
        return self.result_queue.get()
    
    def close(self):
        """
        Safely closes the HDF5 file and stops the loading process.
        """
        if self.loading_process.is_alive():
            self.closing.set()
            self.done_everything.wait()
            self.loading_process.join()
        

    def __del__(self):
        self.close()