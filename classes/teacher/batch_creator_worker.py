from classes.data_classes import Distribution, ConvoTokenized
from numpy import ndarray
import numpy as np
import math


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