import json
import os
import torch
import copy
from time import time
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
from utils.dataset_utils import H5Writer, save_dataset_and_metadata, \
    tokenize_dataset, generate_metadata, save_metadata, \
    good_encode, encode_prompt_format, save_sorted_dataset,\
    get_special_tokens, get_vocab_family, filter_empty_conversations
from utils.convert_to_safetensor import convert_model


def load_model(model_path: str, context_len: int, chunk_size: int, chunk_size_tokens: int):
    print("Loading model...")
    config = ExLlamaV2Config()
    config.model_dir = model_path
    config.prepare()
    config.max_seq_len = context_len
    config.max_batch_size = chunk_size
    config.max_input_len = chunk_size_tokens
    config.max_attention_size = chunk_size_tokens**2

    model = ExLlamaV2(config)
    cache = ExLlamaV2Cache(model, lazy=True)

    model.load_autosplit(cache, reserve_vram=[256 * 2048**2, 128 * 1024**2])
    return model

def is_model_safetensors(model_path: str):
    if os.path.isdir(model_path):
        for file in os.listdir(model_path):
            if file.endswith('.safetensors'):
                return True
        return False
    
@torch.inference_mode()
def run_test_inference(model: ExLlamaV2, prompt_format, model_path, device, text=""):
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    vocab_family = get_vocab_family(tokenizer=tokenizer)
    sp_toks = get_special_tokens(vocab_family)
    text_tokenized = good_encode(text, sp_toks, tokenizer=tokenizer)
    prompt_format_encoded = encode_prompt_format(prompt_format, sp_toks, tokenizer=tokenizer)

    formatted_text = torch.cat((prompt_format_encoded['USER_START'], text_tokenized)).to(device)

    per_token_probs = F.softmax(model.forward(formatted_text.unsqueeze(0)).squeeze(0), dim=1) # type: ignore
    print(per_token_probs[0].size())
    context_token_ids = formatted_text.tolist()
    context_tokens = tokenizer.convert_ids_to_tokens(context_token_ids)
    next_tokens_ids = torch.argmax(per_token_probs, dim=1).tolist()
    next_tokens = tokenizer.convert_ids_to_tokens(next_tokens_ids)

    print(f"Input:\n{context_tokens}\nPredicted Next Tokens:\n{next_tokens}")

@torch.inference_mode()
def eval_ppl(ppl_dataset_path, model, model_path, device, context_len, prompt_format):
    dataset_tokenized = []
    empty_convo_ids = []
    ppl_list = []
    tokens_used = 0
    
    dataset_tokenized, dataset_content_ranges, empty_convo_ids = tokenize_dataset(ppl_dataset_path, device, model_path, prompt_format, context_len, False, False, True, print_stats=False)
    dataset_tokenized, dataset_content_ranges = filter_empty_conversations(dataset_tokenized, dataset_content_ranges, empty_convo_ids)
    pbar = tqdm(zip(dataset_tokenized, dataset_content_ranges), desc="Calculating PPL", unit="convo", smoothing=0.1, leave=False)
    ppl_intermediate = torch.empty(len(dataset_tokenized), device=device)

    for i, (conv_tokenized, conv_content_ranges) in enumerate(pbar):
        conv_tokenized_gpu = conv_tokenized.to(device)[:context_len]
        conv_distributions = F.log_softmax(model.forward(conv_tokenized_gpu.unsqueeze(0)).squeeze(0), dim=1).to(device)

        content_indices = []
        for start, end in conv_content_ranges:
            if start < context_len:
                content_indices.append(torch.arange(start, min(end, context_len), device=device))

        content_indices = torch.cat(content_indices)

        content_distributions = torch.index_select(conv_distributions, 0, content_indices)
        content_tokens = torch.index_select(conv_tokenized_gpu, 0, content_indices)

        gathered_log_probs = torch.gather(content_distributions[:-1], 1, content_tokens[1:].unsqueeze(-1)).squeeze(-1)
        ppl_intermediate[i] = torch.exp((-gathered_log_probs).mean())

        tokens_used += content_distributions.size(0) - 1
        pbar.set_postfix({"Tokens used": tokens_used})
    
    pbar.close()
    ppl_list = ppl_intermediate.to('cpu').tolist()
    ppl = sum(ppl_list) / len(ppl_list)
    print("Preliminary PPL:", round(ppl, 6), "Tokens used:", tokens_used)

@torch.inference_mode()
def generate_probability_distributions(dataset_tokenized, dataset_content_ranges, model: ExLlamaV2, metadata, empty_convo_ids, crop_to_size):
    distribution_size_kb = 0.001953125 * crop_to_size
    approx_size = round((metadata['clean_total_content_tokens'] * distribution_size_kb) / 1000000, 2)
    print(f"Approximate size of the final distributions: {approx_size}GB")

    total_saved_tokens = 0
    eos_id = metadata['eos_id']
    writer = H5Writer(save_folder, timeout=30)
    ppl_intermediate = torch.full((len(dataset_tokenized),), -1, device=device, dtype=torch.float16)
    start_time = time()
    
    def get_content_indices(content_ranges, context_len):
        content_indices = []
        for start, end in content_ranges:
            if start <= context_len:
                content_indices.append(torch.arange(start, min(end, context_len), device=device))
        return torch.cat(content_indices)
    
    pbar = tqdm(zip(dataset_tokenized, dataset_content_ranges), desc="Generating Distributions", unit="convo", total=len(dataset_tokenized), smoothing=0.06)
    for conv_id, (conv_tokenized, conv_content_ranges) in enumerate(pbar):
        if conv_id in empty_convo_ids:
            writer.write_data(torch.tensor([], device=device))
            continue

        conv_tokenized_gpu = conv_tokenized.to(device)[:context_len]
        conv_tokens = conv_tokenized_gpu.numel()

        conv_distributions = model.forward(conv_tokenized_gpu.unsqueeze(0)).squeeze(0) # type: ignore
        conv_distributions = F.log_softmax(conv_distributions, dim=1)
        
        content_indices = get_content_indices(conv_content_ranges, context_len)
        content_distributions = torch.index_select(conv_distributions, 0, content_indices)
        content_tokens = torch.index_select(conv_tokenized_gpu, 0, content_indices)

        gathered_ppl_probs = torch.gather(content_distributions[:-1], 1, content_tokens[1:].unsqueeze(-1)).squeeze(-1)
        ppl_intermediate[conv_id] = torch.exp((-gathered_ppl_probs).mean())

        content_distributions = F.softmax(content_distributions[:, :crop_to_size], dim=1)

        if encourage_eos:
            content_ends = []
            current_id = -1
            for start, end in conv_content_ranges:
                if end > context_len:
                    break
                content_ends.append(((end - start)) + current_id)
                current_id = content_ends[-1]
                    
            for end in content_ends:
                content_distributions[end][eos_id] = torch.max(content_distributions[end]) * 1.1
                content_distributions[end] = content_distributions[end] / content_distributions[end].sum()

        total_saved_tokens += content_distributions.size(0)
        writer.write_data(content_distributions)

        elapsed_time = time() - start_time
        start_time = time()
        tokens_per_second = conv_tokens / elapsed_time if elapsed_time > 0 else 0
        saved_tokens_per_second = content_distributions.size(0) / elapsed_time if elapsed_time > 0 else 0
        pbar.set_postfix({"T/s": round(tokens_per_second, 2), "MB/s": round(saved_tokens_per_second * distribution_size_kb / 1000, 2)})

    writer.close()
    perplexity_list = ppl_intermediate.to('cpu').tolist()

    distributions_metadata = {
        'avg_perplexity': sum(perplexity_list) / len([ppl for ppl in perplexity_list if ppl != -1]),
        'total_saved_tokens': total_saved_tokens,
        'perplexity_list': perplexity_list,
    }
    
    print(f"Total saved tokens: {total_saved_tokens}")
    return distributions_metadata

# Main Script
model_path = r"F:\MythoMax-L2-Kimiko-v2-13b"
dataset_path = r"F:\soup.jsonl"
ppl_dataset_path = r"F:\ppl_test_dataset.jsonl"
distributions_save_folder = r"F:\distilled"
context_len = 2*1024
#batch_size = 4 # How many conversations to process in parallel #TODO
chunk_size = 4 # How many `chunk_size_tokens` chunks to process in parallel
chunk_size_tokens = 1*512

only_get_metadata = False # Won't load the model
test_tokenization = False
test_inference = False
test_ppl = False
ppl_before_start = True
reuse_prompt_format = True

# Model settings
save_sys_range = False
save_user_range = False
save_assistant_range = True
add_bos = True
encourage_eos = True
crop_distr_to_size = 32000
device = "cuda:1"

prompt_format = {
    'SYS_START': "### System:\n",
    'USER_START': "### User:\n",
    'ASSISTANT_START': "### Assistant:\n",
    'SYS_END': '\n',
    'USER_END': '\n',
    'ASSISTANT_END': '\n'
}


pf_status_msg = ""
prompt_format_path = os.path.join(model_path, "prompt_format.json")
if reuse_prompt_format:
    if os.path.exists(prompt_format_path):
        prompt_format = json.load(open(prompt_format_path))
        pf_status_msg = "(Reusing prompt format)"
    else:
        json.dump(prompt_format, open(prompt_format_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)
else:
    json.dump(prompt_format, open(prompt_format_path, 'w', encoding='utf-8'), ensure_ascii=False, indent=4)

model_name = os.path.basename(os.path.normpath(model_path))
dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
save_folder = os.path.join(distributions_save_folder, dataset_name, model_name)

print(f"Model: {model_name} {pf_status_msg}\nDataset: {dataset_name}")

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

config_data = {
    'save_sys_range': save_sys_range,
    'save_user_range': save_user_range,
    'save_assistant_range': save_assistant_range,
    'context_len': context_len,
    'crop_to_size': crop_distr_to_size,
    'prompt_format': copy.deepcopy(prompt_format)
}

if only_get_metadata:
    metadata = {**config_data, **generate_metadata(model_path, [], [])}
    save_sorted_dataset(save_folder, dataset_path)
    save_metadata(metadata, save_folder)
    print("Metadata saved!")
    exit(0)

if test_tokenization:
    dataset_tokenized, dataset_content_ranges, empty_convo_ids = tokenize_dataset(dataset_path, device, model_path, prompt_format, context_len, save_sys_range, save_user_range, save_assistant_range, add_bos=add_bos)
    metadata = {**config_data, **generate_metadata(model_path, dataset_tokenized, dataset_content_ranges, empty_convo_ids=empty_convo_ids, context_len=context_len)}
    save_dataset_and_metadata(dataset_tokenized, dataset_content_ranges, metadata, save_folder)
    exit(0)

if not is_model_safetensors(model_path):
    safetens_path = model_path + "_safetensors"
    if os.path.exists(safetens_path):
        model_path = safetens_path
    else:
        model_path = convert_model(model_path)

model = load_model(model_path, context_len, chunk_size, chunk_size_tokens)

if test_inference:
    run_test_inference(model, prompt_format, model_path, device, text="Hey there, my name is")
    exit(0)

if test_ppl:
    eval_ppl(ppl_dataset_path, model, model_path, device, context_len, prompt_format)
    exit(0)

if ppl_before_start:
    eval_ppl(ppl_dataset_path, model, model_path, device, context_len, prompt_format)

dataset_tokenized, dataset_content_ranges, empty_convo_ids = tokenize_dataset(dataset_path, device, model_path, prompt_format, context_len, save_sys_range, save_user_range, save_assistant_range, add_bos=add_bos)
metadata = {**config_data, **generate_metadata(model_path, dataset_tokenized, dataset_content_ranges, empty_convo_ids=empty_convo_ids, context_len=context_len)}
save_dataset_and_metadata(dataset_tokenized, dataset_content_ranges, metadata, save_folder)

distributions_metadata = generate_probability_distributions(dataset_tokenized, dataset_content_ranges, model, metadata, empty_convo_ids, crop_distr_to_size, )
metadata = {**config_data, **distributions_metadata, **generate_metadata(model_path, dataset_tokenized, dataset_content_ranges, empty_convo_ids=empty_convo_ids, context_len=context_len)}
save_dataset_and_metadata(dataset_tokenized, dataset_content_ranges, metadata, save_folder)
print("Done!")