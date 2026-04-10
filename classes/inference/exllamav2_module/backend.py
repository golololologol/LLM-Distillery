"""ExLlamaV2 inference backend - main process side."""
import queue
import threading
from multiprocessing import get_context
import numpy as np
import torch

from classes.inference.base import InferenceBackend, register_backend
from classes.data_classes import ConvoProcessed
from classes.inference.exllamav2_module.worker import inference_worker
from utils.inference_utils import num_gpus, get_vram_free


class ExLlamaV2Backend(InferenceBackend):
    def __init__(
        self,
        model_path: str,
        context_len: int,
        batch_size: int = 1,
        reserve_vram: list[float] = None,
        seq_chunk_len: int = 256,
        max_queue_size: int = 16,
        marg_chunk_size: int = 256,
        temperature: float = 1.0,
        num_inference_workers: int = 1,
        one_worker_per_gpu: bool = False,
    ):
        self.model_path = model_path
        self.context_len = context_len
        self.batch_size = batch_size
        self.reserve_vram = reserve_vram or []
        self.seq_chunk_len = seq_chunk_len
        self.max_queue_size = max_queue_size
        self.marg_chunk_size = marg_chunk_size
        self.temperature = temperature
        self.one_worker_per_gpu = one_worker_per_gpu
        self.num_inference_workers = num_gpus() if one_worker_per_gpu else num_inference_workers
        self.worker_processes = []
        self.num_alive_workers = 0
        self.inference_queue = None
        self.result_queue = None

    def start(self, dm_queue) -> None:
        self.dm_queue = dm_queue
        ctx = get_context("spawn")
        self.inference_queue = ctx.Queue(self.max_queue_size)
        self.result_queue = ctx.Queue(self.max_queue_size)
        load_status_queue = ctx.Queue()

        gpu_count = num_gpus()
        base_reserve = [256 * 1024**2] * gpu_count
        for i, gb in enumerate(self.reserve_vram):
            if gb > 0 and i < gpu_count:
                base_reserve[i] = int(gb * 1024**3)
        self.worker_processes = []

        worker_args_base = (
            self.inference_queue, self.result_queue, dm_queue, load_status_queue,
            self.model_path, self.context_len, self.batch_size,
        )
        worker_kwargs_tail = (
            self.seq_chunk_len, self.marg_chunk_size, self.model_path, self.temperature,
        )

        if self.one_worker_per_gpu:
            # Parallel loading - each worker gets exclusive GPU
            free_vram = get_vram_free()
            for i in range(min(self.num_inference_workers, gpu_count)):
                gpu_split_gb = [0.0] * gpu_count
                gpu_split_gb[i] = max(0, free_vram[i] - base_reserve[i]) / 1024**3
                process = ctx.Process(
                    target=inference_worker,
                    args=worker_args_base + (gpu_split_gb,) + worker_kwargs_tail + (i,),
                    daemon=True,
                )
                process.start()
                self.worker_processes.append(process)

            for idx in range(len(self.worker_processes)):
                try:
                    status = load_status_queue.get(timeout=600)
                    if status[0] == "ready":
                        if self.num_inference_workers > 1:
                            print(f"  Worker {idx+1}/{len(self.worker_processes)} loaded")
                    else:
                        print(f"  WARNING: A worker failed to load: {status[1]}")
                except queue.Empty:
                    print(f"  WARNING: A worker timed out during loading")

            self.num_alive_workers = sum(1 for p in self.worker_processes if p.is_alive())
        else:
            # Sequential loading with deferred overhead tracking
            deferred_overhead = [0] * gpu_count
            for i in range(self.num_inference_workers):
                free_vram = get_vram_free()
                gpu_split_gb = [
                    max(0, free - reserve - deferred) / 1024**3
                    for free, reserve, deferred in zip(free_vram, base_reserve, deferred_overhead)
                ]
                process = ctx.Process(
                    target=inference_worker,
                    args=worker_args_base + (gpu_split_gb,) + worker_kwargs_tail,
                    daemon=True,
                )
                process.start()

                try:
                    status = load_status_queue.get(timeout=600)
                except queue.Empty:
                    process.terminate()
                    print(f"  WARNING: Worker {i+1} timed out during model loading")
                    continue

                if status[0] == "ready":
                    worker_overhead = status[1]
                    for gpu_idx, overhead_bytes in worker_overhead.items():
                        if gpu_idx < gpu_count:
                            deferred_overhead[gpu_idx] += overhead_bytes
                    self.worker_processes.append(process)
                    if self.num_inference_workers > 1:
                        print(f"  Worker {i+1}/{self.num_inference_workers} loaded")
                else:
                    process.join(timeout=5)
                    if process.is_alive():
                        process.terminate()
                    print(f"  WARNING: Worker {i+1} failed to load: {status[1]}")

            self.num_alive_workers = len(self.worker_processes)
        if self.num_alive_workers == 0:
            raise RuntimeError("No inference workers could start - not enough VRAM")
        if self.num_alive_workers < self.num_inference_workers:
            failed = self.num_inference_workers - self.num_alive_workers
            print(f"  WARNING: {failed} worker(s) could not start due to VRAM constraints, "
                  f"decrease to at least num_inference_workers = {self.num_alive_workers}")

    def process_chunk(
        self,
        samples: list[ConvoProcessed],
        progress_callback: callable = None,
    ) -> None:
        samples = sorted(samples, key=lambda s: s.length, reverse=True)

        batches = []
        for i in range(0, len(samples), self.batch_size):
            batch = samples[i : i + self.batch_size]
            batch_tokens = np.stack([sample.tokens for sample in batch])
            batch_metadata = [
                {
                    "id": sample.origin_convo_id,
                    "content_byte_ranges": sample.content_byte_ranges,
                    "content_sha": sample.content_sha,
                    "cropped": sample.cropped,
                    "length": sample.length,
                }
                for sample in batch
            ]
            batches.append((batch_tokens, batch_metadata))

        def feed_batches():
            for batch in batches:
                self.inference_queue.put(batch)
            for _ in range(self.num_alive_workers):
                self.inference_queue.put(None)

        feeder = threading.Thread(target=feed_batches, daemon=True)
        feeder.start()

        sentinels_received = 0
        try:
            while sentinels_received < self.num_alive_workers:
                try:
                    result = self.result_queue.get(timeout=30)
                except queue.Empty:
                    if not any(p.is_alive() for p in self.worker_processes):
                        break
                    continue
                if result is None:
                    sentinels_received += 1
                elif progress_callback:
                    progress_callback(result)
        except KeyboardInterrupt:
            raise
        finally:
            feeder.join(timeout=2)

    def stop(self) -> None:
        try:
            for p in self.worker_processes:
                if p.is_alive():
                    p.terminate()
            for p in self.worker_processes:
                p.join(timeout=3)
                if p.is_alive():
                    p.kill()
        except (KeyboardInterrupt, OSError):
            for p in self.worker_processes:
                try:
                    p.kill()
                except Exception:
                    pass
        finally:
            self.worker_processes = []
            self.num_alive_workers = 0

            for q in [self.inference_queue, self.result_queue, getattr(self, 'dm_queue', None)]:
                if q is not None:
                    try:
                        q.cancel_join_thread()
                    except Exception:
                        pass
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass


register_backend("exllamav2", ExLlamaV2Backend)
