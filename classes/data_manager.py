from classes.data_classes import Distribution, BatchResult
from multiprocessing import shared_memory
from multiprocessing import get_context
from typing import Any, cast
import multiprocessing
import numpy as np
import hdf5plugin
import traceback
import queue
import time
import h5py
import json
import gc
import os


class DistributionWriteHandle:
    """Pickle-safe write-only handle passed to inference workers."""

    def __init__(self, queue: Any):
        self._queue = queue

    def write_batch(self, batch: list[Distribution]) -> None:
        self._queue.put(('put_batch', batch))

    def cancel_join_thread(self) -> None:
        self._queue.cancel_join_thread()


class H5DataManager:
    def __init__(self, dataset_path, max_queue_size=12, teacher_name: str = "", read_only: bool = False, auto_approve: bool = False):
        self.file_path = os.path.join(dataset_path, f"{teacher_name}.hdf5")
        self.queue: Any = multiprocessing.Queue(max_queue_size)
        self.result_queue: Any = multiprocessing.Queue(max_queue_size)
        self.closing: Any = multiprocessing.Event()
        self.max_queue_size = max_queue_size
        self.loading_process: Any = get_context("spawn").Process(target=self._loading_process)
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
                        ids_to_clear = [self._decode_group_key(group) for group in self._iter_group_names(hdf_file)]
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
                        self.result_queue.put(set([self._decode_group_key(group) for group in self._iter_group_names(hdf_file)]))
                    case 'get_available_shas':
                        self.result_queue.put(self._get_shas(hdf_file))
                    case 'update_shas':
                        self._update_shas(hdf_file, data)
                    case 'get_dataset_attr':
                        self.result_queue.put(self._get_attr(hdf_file, data))
                    case 'set_dataset_attr':
                        self._set_attr(hdf_file, data[0], data[1])
                    case 'has_data':
                        self.result_queue.put(self._has_data(hdf_file))
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
        # h5py attrs can't store dicts or heterogeneous/empty containers natively;
        # encode such values as JSON strings to keep the attribute round-trippable.
        if isinstance(value, dict):
            value = json.dumps(value, sort_keys=True)
        elif isinstance(value, (list, tuple, set, frozenset)):
            seq = list(value)
            if len(seq) == 0 or not all(isinstance(x, (str, bytes, int, float, bool, np.integer, np.floating)) for x in seq):
                value = json.dumps(seq, sort_keys=True, default=str)
        hdf_file.attrs[arg] = value
        
        
    def _group_key(self, convo_id) -> str:
        # h5py treats '/' as a group hierarchy separator, so any convo_id
        # containing '/' (e.g. HuggingFace dataset slugs like
        # "HuggingFaceH4/ultrachat_200k:...") would silently produce nested
        # groups instead of a single flat entry. Escape '/' (and '\\' to keep
        # the encoding reversible) before composing the group name.
        sid = str(convo_id).replace('\\', '\\\\').replace('/', '\\_')
        return f'convo_{sid}'

    @staticmethod
    def _decode_group_key(group_name: str):
        """Inverse of :meth:`_group_key`; returns the original convo_id (int or str)."""
        if not group_name.startswith('convo_'):
            return None
        sid = group_name[len('convo_'):]
        # Reverse the escapes; '\\_' must be undone before '\\\\' to avoid
        # double-decoding a literal backslash followed by underscore.
        out = []
        i = 0
        while i < len(sid):
            if sid[i] == '\\' and i + 1 < len(sid):
                nxt = sid[i + 1]
                if nxt == '_':
                    out.append('/')
                    i += 2
                    continue
                if nxt == '\\':
                    out.append('\\')
                    i += 2
                    continue
            out.append(sid[i])
            i += 1
        decoded = ''.join(out)
        try:
            return int(decoded)
        except ValueError:
            return decoded

    def _iter_group(self, hdf_file: h5py.File):
        return hdf_file

    def _iter_group_names(self, hdf_file: h5py.File) -> list[str]:
        # Only return real top-level convo groups: must start with 'convo_' AND
        # carry the 'content_sha' attribute. This filters out any malformed or
        # legacy nested-parent groups (e.g. files collected before the '/'
        # escape fix where ids containing '/' produced silent group hierarchies).
        return [
            key for key in hdf_file.keys()
            if isinstance(key, str)
            and key.startswith('convo_')
            and 'content_sha' in hdf_file[key].attrs
        ]


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
            ev_shm = distribution.events_from_shd_mem()
            if distribution.distribution is None:
                raise RuntimeError(f"Distribution {distribution.origin_convo_id} has no dense distribution payload")
            self._save_data(
                hdf_file, distribution.distribution, distribution.origin_convo_id,
                distribution.content_sha, cropped=distribution.cropped,
                segment_manifest=distribution.segment_manifest,
                events=distribution.events,
                event_manifest=distribution.event_manifest,
                event_alphabet=distribution.event_alphabet,
                supported_mask=distribution.supported_mask,
                specials_hash=distribution.specials_hash,
            )
            shd_mem.close()
            shd_mem.unlink()
            if ev_shm is not None:
                # release the numpy view first so the buffer is no longer
                # referenced before unmapping the shared segment.
                distribution.events = None
                ev_shm.close()
                ev_shm.unlink()
        

    def _get_shas(self, hdf_file: h5py.File) -> dict:
        shas = {}
        teacher_group = self._iter_group(hdf_file)
        for group in self._iter_group_names(hdf_file):
            shas[self._decode_group_key(group)] = teacher_group[group].attrs['content_sha']
        return shas
    
    def _update_shas(self, hdf_file: h5py.File, shas: dict[int, str]):
        for convo_id, sha in shas.items():
            group_key = self._group_key(convo_id)
            if group_key in hdf_file:
                hdf_file[group_key].attrs['content_sha'] = sha


    def _save_data(
        self,
        hdf_file: h5py.File,
        data: np.ndarray,
        convo_id: int,
        content_sha: str | None = None,
        cropped: bool | None = None,
        segment_manifest=None,
        events: np.ndarray | None = None,
        event_manifest: list[dict] | None = None,
        event_alphabet: tuple[str, ...] | None = None,
        supported_mask: int = 0,
        specials_hash: str | None = None,
    ):
        group_key = self._group_key(convo_id)

        group = cast(h5py.Group, hdf_file.require_group(group_key))

        if content_sha is not None:
            group.attrs['content_sha'] = content_sha
        if cropped is not None:
            group.attrs['cropped'] = cropped
        if segment_manifest is not None:
            group.attrs['segment_manifest'] = json.dumps(segment_manifest)

        zstd = cast(Any, getattr(hdf5plugin, "Zstd"))(clevel=1)
        for key in ['dense_distributions']:
            if key in group:
                del group[key]
        group.create_dataset('dense_distributions', data=data.astype(np.float16), **zstd)
        group.attrs['convo_len'] = data.shape[0]

        # v8 event channel (plan §7).  Always overwrite when the producer
        # supplied any event payload so stale rows from a previous run with a
        # different specials_hash never linger.
        for key in ('event_distributions',):
            if key in group:
                del group[key]
        if events is not None and events.size > 0:
            group.create_dataset('event_distributions', data=events.astype(np.float16), **zstd)
            group.attrs['event_count'] = int(events.shape[0])
        else:
            group.attrs['event_count'] = 0
        if event_manifest is not None:
            group.attrs['event_manifest'] = json.dumps(event_manifest)
        if event_alphabet is not None:
            # Store as a list/ndarray so h5py serialises one string per slot;
            # JSON-encoding the whole list character-splits on read and breaks
            # cross-teacher merging (the merger compares alphabets element-wise).
            group.attrs['event_alphabet'] = list(event_alphabet)
        if supported_mask:
            group.attrs['supported_mask'] = int(supported_mask)
        if specials_hash is not None:
            group.attrs['specials_hash'] = specials_hash
            # Mirror at the file level so loaders can reject a mixed cache
            # without iterating every group.
            hdf_file.attrs['specials_hash'] = specials_hash
        if event_alphabet is not None:
            hdf_file.attrs['event_alphabet'] = list(event_alphabet)
        # Bump file-level layout version once we are writing v8 fields.
        if events is not None or event_manifest is not None or specials_hash is not None:
            hdf_file.attrs['layout_version'] = 6
    
    
    def _load_group_distributions(self, group):
        if 'dense_distributions' not in group:
            return None
        return np.array(group['dense_distributions'][:], dtype=np.float32)


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


    def _clear_dataset(self, hdf_file: h5py.File, ids_to_clear: list[int] | None = None):
        for id in (ids_to_clear or []):
            try:
                del hdf_file[self._group_key(id)]
            except:
                pass

        self.shared_batches = []


    def _rename_ids(self, hdf_file: h5py.File, ids_to_reindex: dict[int, int]):
        for old_id, new_id in ids_to_reindex.items():
            hdf_file.move(self._group_key(old_id), self._group_key(new_id) + '_moved')

        for old_id, new_id in ids_to_reindex.items():
            hdf_file.move(self._group_key(new_id) + '_moved', self._group_key(new_id))
            
    
    def enqueue_get_batches(self, batches: list[list[int]]):
        self.queue.put(('get_batches', batches))

    @property
    def writer(self) -> DistributionWriteHandle:
        return DistributionWriteHandle(self.queue)

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
    
    def delete_ids(self, ids: list[int], reason: str | None = None):
        if not ids:
            return
        
        if reason:
            msg = f"{reason}\nThis will delete {len(ids)} samples."
        else:
            msg = f"Deleting {len(ids)} samples."
        
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
        
        msg = f"Renaming {len(ids_to_reindex)} samples to sync IDs with text dataset."
        
        if self.auto_approve:
            print(f"WARNING: {msg} (auto-approved)")
        else:
            response = input(f"{msg}\nProceed? (y/n): ")
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

    def get_dataset_attr(self, attr: str):
        self.queue.put(('get_dataset_attr', attr))
        return self.result_queue.get()
    
    def has_data(self) -> bool:
        self.queue.put(('has_data', None))
        self._flush_and_wait()
        return self.result_queue.get()
    
    def close(self):
        try:
            loading_process = getattr(self, "loading_process", None)
            if loading_process is None or not loading_process.is_alive():
                return
            self.closing.set()
            try:
                self.queue.put(None, timeout=2)
            except (queue.Full, OSError):
                pass
            loading_process.join(timeout=5)
            if loading_process.is_alive():
                loading_process.terminate()
                loading_process.join(timeout=2)
        except (OSError, ValueError, KeyboardInterrupt, InterruptedError):
            if loading_process is not None:
                try:
                    loading_process.terminate()
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
