from classes.data_classes import Distribution, BatchResult
from multiprocessing import shared_memory
from multiprocessing import get_context
import multiprocessing
import numpy as np
import traceback
import queue
import time
import h5py
import gc
import os


def _compress_distributions(data):
    """Per row: [count:u16][indices:u8[count]][values:f16[count]]"""
    data = np.asarray(data, dtype=np.float32)
    num_rows = data.shape[0]
    offsets = np.empty(num_rows, dtype=np.uint64)
    parts = []
    byte_pos = 0

    for i in range(num_rows):
        row = data[i]
        nonzero = row > 0
        nonzero_count = int(nonzero.sum())
        offsets[i] = byte_pos

        indices = np.where(nonzero)[0].astype(np.uint8)
        values = row[nonzero].astype(np.float16)
        packed_row = nonzero_count.to_bytes(2, 'little') + indices.tobytes() + values.tobytes()

        parts.append(packed_row)
        byte_pos += len(packed_row)

    return b''.join(parts), offsets


def _decompress_distributions(packed_data, offsets, convo_len):
    buf = np.frombuffer(packed_data, dtype=np.uint8)
    offsets = offsets.astype(np.int64)
    result = np.zeros((convo_len, 256), dtype=np.float32)

    counts = buf[offsets].astype(np.uint16) | (buf[offsets + 1].astype(np.uint16) << 8)
    has_entries = counts > 0
    if not has_entries.any():
        return result

    active_rows = np.where(has_entries)[0]
    active_counts = counts[has_entries].astype(np.int64)
    active_offsets = offsets[has_entries] + 2  # skip the 2-byte count header
    total_entries = int(active_counts.sum())

    # Build flat element positions: which byte in the buffer holds each index and value
    count_cumsum = np.empty(len(active_rows) + 1, dtype=np.int64)
    count_cumsum[0] = 0
    np.cumsum(active_counts, out=count_cumsum[1:])

    element_offset = np.arange(total_entries, dtype=np.int64) - np.repeat(count_cumsum[:-1], active_counts)
    index_positions = np.repeat(active_offsets, active_counts) + element_offset
    value_positions = np.repeat(active_offsets + active_counts, active_counts) + element_offset * 2

    all_indices = buf[index_positions]
    raw_f16 = buf[value_positions].astype(np.uint16) | (buf[value_positions + 1].astype(np.uint16) << 8)
    all_values = raw_f16.view(np.float16).astype(np.float32)

    result[np.repeat(active_rows, active_counts), all_indices] = all_values
    return result


class H5DataManager:
    def __init__(self, dataset_path, max_queue_size=12, teacher_name: str = "", read_only: bool = False, auto_approve: bool = False):
        self.file_path = os.path.join(dataset_path, "distributions.hdf5")
        self.queue = multiprocessing.Queue(max_queue_size)
        self.result_queue = multiprocessing.Queue(max_queue_size)
        self.closing = multiprocessing.Event()
        self.max_queue_size = max_queue_size
        self.loading_process = get_context("spawn").Process(target=self._loading_process)
        self.shared_batches: list[shared_memory.SharedMemory] = []
        self.teacher_name = teacher_name
        self.read_only = read_only
        self.auto_approve = auto_approve

        self.loading_process.start()

    def _loading_process(self):
        import signal
        def _interrupt_handler(signum, frame):
            self.closing.set()
        signal.signal(signal.SIGINT, _interrupt_handler)
        signal.signal(signal.SIGTERM, _interrupt_handler)

        def handle_exit(signum, frame):
            global hdf_file
            if hdf_file is not None:
                if not self.read_only:
                    hdf_file.flush()
                hdf_file.close()

            if self.queue is not None:
                self._clear_queue()
                self.queue.cancel_join_thread()

            if self.result_queue is not None:
                self._clear_result_queue()
                self.result_queue.cancel_join_thread()

            for shared_batch in self.shared_batches:
                shared_batch.close()
                shared_batch.unlink()
                
            self.shared_batches = []
            hdf_file = None
            
            gc.collect()

        global hdf_file
        hdf_file = h5py.File(self.file_path, 'r' if self.read_only else 'a')

        try:
            while True:
                try:
                    item = self.queue.get(timeout=1.0)
                except (queue.Empty, KeyboardInterrupt, InterruptedError, OSError):
                    if self.closing.is_set():
                        break
                    continue
                if item is None:
                    break

                task, data = item

                match task:
                    case 'get_batch':
                        self.result_queue.put(self._make_outgoing_batch(hdf_file, data))
                    case 'get_batches':
                        self._get_batches(hdf_file, data)
                    case 'read_only_mode':
                        self._read_only_mode(hdf_file, data)
                    case 'put_batch':
                        self._process_distributions(hdf_file, data)
                    case 'clear_dataset':
                        ids_to_clear = [int(group.split('_')[1]) for group in self._iter_group(hdf_file)]
                        self._clear_dataset(hdf_file, ids_to_clear)
                    case 'clear_queues':
                        self._clear_queues()
                    case 'clear_queue':
                        self._clear_queue()
                    case 'clear_result_queue':
                        self._clear_result_queue()
                    case 'clear_ids':
                        self._clear_dataset(hdf_file, data)
                    case 'rename_ids':
                        self._rename_ids(hdf_file, data)
                    case 'get_available_ids':
                        self.result_queue.put(set([int(group.split('_')[1]) for group in self._iter_group(hdf_file)]))
                    case 'get_available_shas':
                        self.result_queue.put(self._get_shas(hdf_file))
                    case 'update_shas':
                        self._update_shas(hdf_file, data)
                    case 'get_dataset_attr':
                        self.result_queue.put(self._get_attr(hdf_file, data))
                    case 'set_dataset_attr':
                        self._set_attr(hdf_file, data[0], data[1])
                    case 'get_teacher_attr':
                        self.result_queue.put(self._get_teacher_attr(hdf_file, data))
                    case 'set_teacher_attr':
                        self._set_teacher_attr(hdf_file, data)
                    case 'has_data':
                        self.result_queue.put(self._has_data(hdf_file))
                    case 'merge_teachers':
                        self._merge_teachers(hdf_file, data)
                    case '_flush':
                        data.send(True)
                        data.close()
                    case _:
                        print(f"[WARN] H5DataManager: unknown task '{task}'")

        except (KeyboardInterrupt, InterruptedError, OSError):
            pass
        except Exception as e:
            print(f"Data Manager process exception: {e}")
            traceback.print_exc()
        finally:
            handle_exit(None, None)


    def _get_attr(self, hdf_file: h5py.File, arg):
        return hdf_file.attrs.get(arg, None)
    
    def _set_attr(self, hdf_file: h5py.File, arg, value):
        hdf_file.attrs[arg] = value
        
        
    def _group_key(self, convo_id: int) -> str:
        if self.teacher_name:
            return f'{self.teacher_name}/convo_{convo_id}'
        return f'convo_{convo_id}'

    def _iter_group(self, hdf_file: h5py.File):
        if self.teacher_name:
            if self.teacher_name in hdf_file:
                return hdf_file[self.teacher_name]
            return {}
        return hdf_file
        
        
    def _get_teacher_attr(self, hdf_file: h5py.File, arg):
        if self.teacher_name in hdf_file:
            return hdf_file[self.teacher_name].attrs.get(arg, None)

    def _set_teacher_attr(self, hdf_file: h5py.File, data):
        attr_name, value = data
        group = hdf_file.require_group(self.teacher_name)
        group.attrs[attr_name] = value


    def _has_data(self, hdf_file: h5py.File):
        return len(self._iter_group(hdf_file)) > 0
    

    def _get_batches(self, hdf_file, data):
        for batch_ids in data:
            if self.closing.is_set():
                return
            batch = self._make_outgoing_batch(hdf_file, batch_ids)
            while not self.closing.is_set():
                try:
                    self.result_queue.put(batch, timeout=1.0)
                    break
                except (queue.Full, InterruptedError, OSError):
                    continue


    def _read_only_mode(self, hdf_file, data):
        while not self.closing.is_set():
            for batch_ids in data:
                if self.closing.is_set():
                    return
                batch = self._make_outgoing_batch(hdf_file, batch_ids)
                while not self.closing.is_set():
                    try:
                        self.result_queue.put(batch, timeout=1.0)
                        break
                    except (queue.Full, InterruptedError, OSError):
                        continue
        
 
    def _process_distributions(self, hdf_file: h5py.File, batch: list[Distribution]):
        for distribution in batch:
            shd_mem = distribution.from_shd_mem()
            self._save_data(hdf_file, distribution.distribution, distribution.origin_convo_id, distribution.content_sha, cropped=distribution.cropped)
            shd_mem.close()
            shd_mem.unlink()
        

    def _get_shas(self, hdf_file: h5py.File) -> dict[int, str]:
        shas = {}
        teacher_group = self._iter_group(hdf_file)
        for group in teacher_group:
            shas[int(group.split('_')[1])] = teacher_group[group].attrs['content_sha']
        return shas
    
    def _update_shas(self, hdf_file: h5py.File, shas: dict[int, str]):
        for id, sha in shas.items():
            group_key = self._group_key(id)
            if group_key in hdf_file:
                hdf_file[group_key].attrs['content_sha'] = sha


    def _save_data(self, hdf_file: h5py.File, data: np.ndarray, convo_id: int, content_sha: str = None, cropped: bool = None):
        group_key = self._group_key(convo_id)

        if group_key not in hdf_file:
            group = hdf_file.create_group(group_key)
        else:
            group = hdf_file[group_key]

        if content_sha is not None:
            group.attrs['content_sha'] = content_sha
        if cropped is not None:
            group.attrs['cropped'] = cropped

        packed_bytes, offsets = _compress_distributions(data)
        for key in ['distributions_data', 'distributions_offsets']:
            if key in group:
                del group[key]
        group.create_dataset('distributions_data', data=np.frombuffer(packed_bytes, dtype=np.uint8), compression='gzip', compression_opts=1)
        group.create_dataset('distributions_offsets', data=offsets, compression='gzip', compression_opts=1)
        group.attrs['convo_len'] = data.shape[0]
    
    
    def _load_group_distributions(self, group):
        if 'distributions_data' not in group:
            return None
        packed_data = bytes(group['distributions_data'][:])
        offsets = np.array(group['distributions_offsets'][:])
        convo_len = int(group.attrs['convo_len'])
        return _decompress_distributions(packed_data, offsets, convo_len)


    def _load_id(self, hdf_file: h5py.File, convo_id: int) -> np.ndarray:
        group_key = self._group_key(convo_id)
    
        if group_key not in hdf_file:
            raise ValueError(f"Convo ID {convo_id} not found in dataset.")
        
        group = hdf_file[group_key]
        result = self._load_group_distributions(group)
        if result is None:
            raise ValueError(f"Convo ID {convo_id} has no distributions in dataset.")
        return result
    

    def _make_outgoing_batch(self, hdf_file: h5py.File, batch_ids: list[int]) -> tuple[str, tuple[int, int], np.dtype, list[int]]:
        batch = []
        for convo_id in batch_ids:
            batch.append(self._load_id(hdf_file, convo_id))

        max_len = max(len(distr) for distr in batch)
        dtype = batch[0].dtype
        shape = (len(batch_ids), max_len, 256)

        shared_batch_memory = shared_memory.SharedMemory(create=True, size=int(np.prod(shape)) * dtype.itemsize)
        shared_batch = np.ndarray(shape, dtype=dtype, buffer=shared_batch_memory.buf)
        shared_batch[:] = 0

        batch_padding = []
        for i, distr in enumerate(batch):
            shared_batch[i, :len(distr)] = distr
            batch_padding.append(len(distr))

        self.shared_batches.append(shared_batch_memory)

        if len(self.shared_batches) >= self.max_queue_size + 10:
            self.shared_batches[0].close()
            self.shared_batches[0].unlink()
            self.shared_batches = self.shared_batches[1:]

        return (shared_batch_memory.name, shared_batch.shape, shared_batch.dtype, batch_padding)


    def _clear_queues(self):
        self._clear_queue()
        self._clear_result_queue()
        
    def _clear_result_queue(self):
        while not self.result_queue.empty():
            try:
                self.result_queue.get_nowait()
            except Exception:
                break
        self.shared_batches = []
    
    def _clear_queue(self):
        while not self.queue.empty():
            try:
                self.queue.get_nowait()
            except Exception:
                break


    def _clear_dataset(self, hdf_file: h5py.File, ids_to_clear: list[int] = None):
        for id in (ids_to_clear or []):
            try:
                del hdf_file[self._group_key(id)]
            except:
                pass

        self.shared_batches = []


    def _merge_teachers(self, hdf_file: h5py.File, teacher_weights: dict[str, float]):
        teacher_names = list(teacher_weights.keys())

        first_teacher = teacher_names[0]
        if first_teacher not in hdf_file:
            return

        convo_ids = [int(convo.split('_')[1]) for convo in hdf_file[first_teacher]]

        if '_merged' in hdf_file:
            del hdf_file['_merged']
        merged = hdf_file.create_group('_merged')

        for id in convo_ids:
            teacher_convos = []
            max_len = 0
            
            for teacher in teacher_names:
                teacher_convo_id = f'{teacher}/convo_{id}'
                if teacher_convo_id in hdf_file:
                    distr = self._load_group_distributions(hdf_file[teacher_convo_id])
                    if distr is not None:
                        teacher_convos.append((distr, distr.shape[0], teacher_weights[teacher]))
                        max_len = max(max_len, distr.shape[0])

            if not teacher_convos:
                continue

            merged_distr = np.zeros((max_len, 256), dtype=np.float32)
            total_weights = np.zeros(max_len, dtype=np.float32)
            for distr, length, weight in teacher_convos:
                merged_distr[:length] += weight * distr
                total_weights[:length] += weight
            has_data = total_weights > 0
            merged_distr[has_data] /= total_weights[has_data, np.newaxis]

            sha = hdf_file[f'{teacher_names[0]}/convo_{id}'].attrs.get('content_sha', '')
            merged_convo = merged.create_group(f'convo_{id}')
            packed_bytes, offsets = _compress_distributions(merged_distr)
            merged_convo.create_dataset('distributions_data', data=np.frombuffer(packed_bytes, dtype=np.uint8), compression='gzip', compression_opts=1)
            merged_convo.create_dataset('distributions_offsets', data=offsets, compression='gzip', compression_opts=1)
            merged_convo.attrs['content_sha'] = sha
            merged_convo.attrs['convo_len'] = merged_distr.shape[0]

    def _rename_ids(self, hdf_file: h5py.File, ids_to_reindex: dict[int, int]):
        for old_id, new_id in ids_to_reindex.items():
            hdf_file.move(self._group_key(old_id), self._group_key(new_id) + '_moved')

        for old_id, new_id in ids_to_reindex.items():
            hdf_file.move(self._group_key(new_id) + '_moved', self._group_key(new_id))
            
    
    def enqueue_get_batches(self, batches: list[list[int]]):
        self.queue.put(('get_batches', batches))

    def _flush_and_wait(self):
        r, w = multiprocessing.Pipe(duplex=False)
        self.queue.put(('_flush', w))
        r.recv()
        r.close()
        w.close()

    def read_next_batch(self):
        shm_name, shape, dtype, padding = self.result_queue.get()
        return BatchResult(shm_name, shape, dtype, padding)

    def read_only_mode(self, batches: list[list[int]]):
        self.queue.put(('read_only_mode', batches))

    def write_batch(self, batch: list[Distribution]):
        self.queue.put(('put_batch', batch))

    def get_dataset_ids(self) -> set[int]:
        self.queue.put(('get_available_ids', None))
        return self.result_queue.get()

    def purge_dataset(self, ask_confirmation=True):
        true_replies = ['y', 'yes', 'ye', '1', 'true', 't']

        if self.auto_approve:
            print("WARNING: Deleting all distributions from the h5 dataset. (auto-approved)")
        elif ask_confirmation:
            reply = input("The script is going to delete all distributions from the h5 dataset.\nAre you sure you want to proceed? (y/n): ")
            if reply.lower() not in true_replies:
                raise ValueError("User cancelled operation.")
        
            reply = input("Are you REALLY sure you want to delete all distributions from the dataset? (y/n): ")
            if reply.lower() not in true_replies:
                raise ValueError("User cancelled operation.")
        
        self.queue.put(('clear_dataset', None))
        self._flush_and_wait()
    
    def delete_ids(self, ids: list[int], reason: str = None):
        if not ids:
            return
        
        label = f"{self.teacher_name} " if self.teacher_name else ""
        if reason:
            msg = f"{reason}\nThis will delete {len(ids)} samples from the {label}h5 dataset."
        else:
            msg = f"The script called for deletion of {len(ids)} samples from the {label}h5 dataset."
        
        if self.auto_approve:
            print(f"WARNING: {msg} (auto-approved)")
        else:
            response = input(f"{msg}\nProceed? (y/n): ")
            if response.lower() not in ['y', 'yes', 'ye', '1', 'true', 't']:
                raise ValueError("User cancelled operation.")
        
        self.queue.put(('clear_ids', ids))
        self._flush_and_wait()

    def rename_ids(self, ids_to_reindex: dict[int, int]):
        if not ids_to_reindex:
            return
        
        label = f"{self.teacher_name} " if self.teacher_name else ""
        msg = f"The script called for renaming of {len(ids_to_reindex)} samples in the {label}h5 dataset.\nThis means that the dataset's samples will be moved to new IDs to be in sync with your current text dataset."
        
        if self.auto_approve:
            print(f"WARNING: {msg} (auto-approved)")
        else:
            response = input(f"{msg}\nAre you sure you want to proceed? (y/n): ")
            if response.lower() not in ['y', 'yes', 'ye', '1', 'true', 't']:
                raise ValueError("User cancelled operation.")
        
        self.queue.put(('rename_ids', ids_to_reindex))
        self._flush_and_wait()

    def sync(self, ids_to_delete, ids_to_reindex):
        self._flush_and_wait()
        self.delete_ids(ids_to_delete)

        self._flush_and_wait()
        self.rename_ids(ids_to_reindex)
        
        self._flush_and_wait()

    def get_available_shas(self) -> dict[int, str]:
        self._flush_and_wait()
        self.queue.put(('get_available_shas', None))
        return self.result_queue.get()
    
    def update_shas(self, shas: dict[int, str]):
        self.queue.put(('update_shas', shas))
        self._flush_and_wait()

    def set_dataset_attr(self, attr: str, value):
        self.queue.put(('set_dataset_attr', (attr, value)))

    def set_teacher_attr(self, attr: str, value):
        self.queue.put(('set_teacher_attr', (attr, value)))

    def get_teacher_attr(self, attr: str):
        self.queue.put(('get_teacher_attr', attr))
        return self.result_queue.get()

    def get_dataset_attr(self, attr: str):
        self.queue.put(('get_dataset_attr', attr))
        return self.result_queue.get()
    
    def has_data(self) -> bool:
        self.queue.put(('has_data', None))
        self._flush_and_wait()
        return self.result_queue.get()
    
    def merge_teachers(self, teacher_weights: dict[str, float]):
        self.queue.put(('merge_teachers', teacher_weights))
        self._flush_and_wait()

    def close(self):
        try:
            if not self.loading_process.is_alive():
                return
            self.closing.set()
            try:
                self.queue.put(None, timeout=2)
            except (queue.Full, OSError):
                pass
            self.loading_process.join(timeout=5)
            if self.loading_process.is_alive():
                self.loading_process.terminate()
                self.loading_process.join(timeout=2)
        except (OSError, ValueError, KeyboardInterrupt, InterruptedError):
            try:
                self.loading_process.terminate()
            except Exception:
                pass
        finally:
            for q in [self.queue, self.result_queue]:
                if q is not None:
                    try:
                        q.cancel_join_thread()
                    except Exception:
                        pass

    def __del__(self):
        self.close()