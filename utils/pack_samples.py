import json
import os
from exllamav2 import ExLlamaV2Tokenizer, ExLlamaV2Config
import numpy as np
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import defaultdict
from tqdm import tqdm

def good_encode(id: int, text: str, sp_toks: dict, tokenizer, encode_special=True, replace_tokens=True):
    if replace_tokens:
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