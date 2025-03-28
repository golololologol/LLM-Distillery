from joblib import Parallel, delayed
import numpy as np



def _process_logprobs_dict(prob_dict, topK, include_sampled):
    """
    Process a dictionary of topK token logprobs and indices into NumPy arrays.

    It selects the topK tokens based on their ranking.
    If an extra sampled token is present and `include_sampled` is True,
    the candidate with the lowest log probability is replaced by this sampled token.

    Args:
        prob_dict (dict): A dictionary where keys are token IDs and values are objects 
            containing attributes `logprob` and `rank`. An empty dictionary will result in empty arrays.
        topK (int): The number of top tokens to select.
        include_sampled (bool): Whether to include a sampled token by replacing the candidate 
            with the worst log probability if the dictionary has an extra element.

    Returns:
        tuple: A tuple containing two numpy.ndarray objects:
            - The first array holds the log probability values for the selected tokens.
            - The second array holds the corresponding token IDs.
    """
    if not prob_dict:
        return np.array([]), np.array([])

    # Convert dictionary items to a list of tuples (token_id, logprob_object) and sort by rank.
    items = list(prob_dict.items())
    items.sort(key=lambda x: x[1].rank)

    if len(items) == topK:
        # No extra sampled token, use the provided tokens.
        candidates = items
    else:
        # The last element is the sampled token (with rank > topK)
        sampled = items[-1]
        candidates = items[:topK]
        if include_sampled:
            # Replace the worst candidate with the sampled token.
            candidates[-1] = sampled

    # Extract logprob values and token IDs.
    values = [entry[1].logprob for entry in candidates]
    indices = [entry[0] for entry in candidates]
    return np.array(values), np.array(indices)


def gather_aphrodite_logprobs(request_outputs, topK=3, include_sampled=False, worker_pool=None):
    """
    Gather token logprobs and indices from multiple request outputs from Aphrodite
    and return the results as stacked NumPy arrays.

    This function iterates over a collection of output objects, each containing 
    log probability information for both prompt tokens and the final predicted token. 
    It flattens these tokens into a list, processes each using `_process_logprobs_dict`
    in parallel, and then stacks the resulting arrays to produce aggregated log 
    probability values and token indices.

    Args:
        request_outputs (iterable): A collection of output objects, where each object 
            is expected to have:
                - `prompt_logprobs`: a list where the first element is always None and 
                  subsequent elements are token log probability dictionaries.
                - `outputs`: a list whose first element contains a `logprobs` attribute, 
                  which is itself a list (the first element is used).
        topK (int, optional): The number of top candidate tokens to select from each token's 
            log probabilities. Defaults to 3.
        include_sampled (bool, optional): If True, includes a sampled token in the candidates 
            by replacing the candidate with the worst log probability. Defaults to False.
        worker_pool (Parallel, optional): A parallel worker pool for processing the tokens.
            If None, a new Parallel pool using all available cores with a threaded backend is created.

    Returns:
        dict: A dictionary with two keys:
            - "values": A numpy.ndarray containing the stacked log probability values for each token.
            - "indices": A numpy.ndarray containing the corresponding token IDs.
    """
    tokens_to_process = []

    # Flatten all valid token logprobs from all outputs.
    for output in request_outputs:
        # Skip the first token in prompt_logprobs as it is always None.
        tokens_to_process.extend(output.prompt_logprobs[1:])
        # Append the final output token predictions.
        tokens_to_process.append(output.outputs[0].logprobs[0])

    # Use the provided worker pool or create a new one.
    if worker_pool is None:
        worker_pool = Parallel(n_jobs=-1, backend="threading")

    results = worker_pool(
        delayed(_process_logprobs_dict)(token, topK, include_sampled)
        for token in tokens_to_process
    )

    # Unzip the results into separate lists for values and indices.
    values_list, indices_list = zip(*results)

    return {
        "values": np.stack(values_list, axis=0),
        "indices": np.stack(indices_list, axis=0)
    }