"""
This tool optimizes training data for language models by packing multiple
shorter samples into combined samples to maximize context window utilization.

Purpose:
- Processes JSONL training data files containing conversation samples
- Tokenizes samples to determine their exact token lengths
- Efficiently packs multiple short samples together to fill the context window
- Preserves proper separation between samples using BOS/EOS tokens
- Outputs a new JSONL file with the packed samples sorted by length

The packing algorithm prioritizes fitting the biggest samples possible into the
available space without exceeding it, which very efficiently utilizes the
available samples for packing.
"""

import json
import os
from exllamav2.config import ExLlamaV2Config
from exllamav2.tokenizer.tokenizer import ExLlamaV2Tokenizer
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm

def good_encode(id: int | None, text: str, sp_toks: dict | None, tokenizer, encode_special=True, replace_tokens=True):
    if replace_tokens:
        if sp_toks is None:
            raise ValueError("sp_toks is required when replace_tokens=True")
        text = text.replace('<bos>', sp_toks["bos"]).replace('<eos>', sp_toks["eos"])

    if tokenizer.__class__.__name__ != "ExLlamaV2Tokenizer":
        encoded_ids = tokenizer.encode("\n" + text, add_special_tokens=False)[2:]
    else:
        encoded_ids = tokenizer.encode("\n" + text, encode_special_tokens=encode_special).squeeze(0)[2:] # type: ignore

    encoded_text = np.array(encoded_ids, dtype=np.int64)

    if id is None:
        return encoded_text
    
    return id, encoded_text

def write_samples(samples, outfile, sort):
    if sort:
        samples.sort(key=lambda x: x['encoded_len'], reverse=True)
    for sample in samples:
        outfile.write(json.dumps(sample, ensure_ascii=False) + '\n')

def find_longest_fitting_len(current_len, candidates, context_len, special_tokens_len):
    combined_lengths = current_len + special_tokens_len + np.array(candidates)
    valid_indices = np.where(combined_lengths <= context_len)[0]
    return int(valid_indices[-1]) if len(valid_indices) > 0 else None

def prepack(input_path, output_path, model_path, context_len, min_desired_len, bos, eos):
    """
        Pack conversation samples to optimize context window utilization.
        This function processes a JSONL file containing conversation samples, encodes them
        using the specified tokenizer, and packs multiple shorter conversations together
        to better utilize the context length during training. The packing algorithm tries
        to combine samples to approach but not exceed the context length.
        Args:
            input_path (str): Path to the input JSONL file containing conversation samples
            output_path (str): Path where packed samples will be written as JSONL
            model_path (str): Path to the ExLlamaV2 model for tokenization
            context_len (int): Maximum allowed context length in tokens
            min_desired_len (int): Target minimum length for packed sequences
            bos (str): Beginning of sequence token/string to use between packed samples
            eos (str): End of sequence token/string to use between packed samples
        Returns:
            None: Results are written to the output_path file
        Note:
            Each input sample is expected to be a JSON object with a 'conversations' field
            containing an array, where the first element is the conversation text.
    """
    config = ExLlamaV2Config()
    config.model_dir = model_path
    config.prepare()
    tokenizer = ExLlamaV2Tokenizer(config)
    bos_len = len(good_encode(None, bos, None, tokenizer, replace_tokens=False))
    eos_len = len(good_encode(None, eos, None, tokenizer, replace_tokens=False))
    special_tokens_len = bos_len + eos_len
    
    with open(input_path, 'r', encoding='utf-8') as infile, open(output_path, 'w', encoding='utf-8') as outfile:
        dataset = [json.loads(line) for line in infile]

        def generate_tasks():
            for id, sample in enumerate(dataset):
                yield id, sample["conversations"][0]

        with ThreadPoolExecutor(max_workers=8) as executor:
            pbar = tqdm(total=len(dataset), desc="Encoding", unit="samples", smoothing=0.06)
            futures = {executor.submit(good_encode, id, convo, None, tokenizer, False, False): (id, convo) for id, convo in generate_tasks()}
            for future in as_completed(futures):
                id, encoded = future.result()
                dataset[id]['encoded_len'] = len(encoded)
                pbar.update(1)
            pbar.close()
        dataset.sort(key=lambda x: x['encoded_len'])
        grouped_dict = defaultdict(list)
        lens = [sample['encoded_len'] for sample in dataset]
        packed_lens = []
        for sample in dataset:
            grouped_dict[sample['encoded_len']].append(sample)

        lens = np.array(lens, dtype=int)
        pbar = tqdm(total=len(dataset), desc="Calculating packing", unit="samples", smoothing=0.06)
        while len(lens) > 0:
            current_len = lens[0]
            lens = np.delete(lens, 0)
            group_lens = [int(current_len)]
        
            if current_len > context_len:
                packed_lens.append(group_lens)
                continue
            
            combined_len = current_len
            while True:
                if combined_len >= min_desired_len:
                    break
                best_fit = find_longest_fitting_len(combined_len, lens, context_len, special_tokens_len)
                if best_fit is None:
                    break
                best_fit_len = lens[best_fit]
                lens = np.delete(lens, best_fit)
                combined_len += special_tokens_len + best_fit_len
                group_lens.append(int(best_fit_len))

            packed_lens.append(group_lens)
            pbar.total = len(lens) + len(packed_lens)
            pbar.update()

        pbar.close()

        samples = []
        for group_lens in tqdm(packed_lens, desc="Packing", unit="groups", smoothing=0.06):
            base_key = group_lens.pop(0)
            base_sample = grouped_dict[base_key].pop()
            for candidate_len in group_lens:
                candidate_sample = grouped_dict[candidate_len].pop()
                base_sample['conversations'][0] += eos + bos + candidate_sample['conversations'][0]
                base_sample['encoded_len'] += special_tokens_len + candidate_len
            samples.append(base_sample)

        write_samples(samples, outfile, sort=True)



input_file_path = r"C:\Users\PC\Converted_random_samples_200k.jsonl"
model_path = r"C:\Users\PC\Desktop\LLaMA2-13B-Tiefighter_safetensors"
context_len = 8*1024
min_desired_len = 7500
bos = "<bos>"
eos = "<eos>"
path = os.path.dirname(input_file_path)
name = os.path.basename(input_file_path).split('.')[0]
output_file_path = os.path.join(path, f"Prepacked_{name}.jsonl")
dataset_name = name

prepack(input_file_path, output_file_path, model_path, context_len, min_desired_len, bos, eos)
print(f"Prepacked {input_file_path} to {output_file_path} with dataset name {dataset_name}")