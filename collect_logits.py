import os
import threading
import pickle
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
from utils.dataset_utils import save_dataset_and_metadata, \
    tokenize_dataset, generate_metadata, save_metadata, \
    good_encode, encode_prompt_format, save_sorted_dataset
from utils.finetuning_utils import create_mask, preprocess_logits

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

    model.load_autosplit(cache)
    return model

def async_save_partial_distributions(dataset_distributions: list, count: int):
    save_path = os.path.join(save_folder, "distributions")
    dataset_distributions_cpu = []
    for convo_distr in dataset_distributions:
        dataset_distributions_cpu.append(convo_distr.numpy())
        
    def save_thread():
        with open(f'{save_path}_{count}.pkl', 'wb') as f:
            pickle.dump(dataset_distributions_cpu, f)

    thread = threading.Thread(target=save_thread)
    thread.start()

def run_test_inference(model: ExLlamaV2, prompt_format, model_path, device, text=""):
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    text_tokenized = good_encode(text, tokenizer=tokenizer)
    prompt_format_encoded = encode_prompt_format(prompt_format, tokenizer=tokenizer)

    formatted_text = torch.cat((prompt_format_encoded['USER_START'], text_tokenized)).to(device)

    per_token_logits = model.forward(formatted_text.unsqueeze(0)).squeeze(0) # type: ignore

    context_token_ids = formatted_text.tolist()
    context_tokens = tokenizer.convert_ids_to_tokens(context_token_ids)
    next_tokens_ids = torch.argmax(per_token_logits, dim=1).tolist()
    next_tokens = tokenizer.convert_ids_to_tokens(next_tokens_ids)

    print(f"Input:\n{context_tokens}\nPredicted Next Tokens:\n{next_tokens}")


def generate_probability_distributions(dataset_tokenized, dataset_content_ranges, model, device, max_cache_gb, next_token_prob_boost, context_len, metadata, empty_convo_ids):
    TOKEN_DISTRIBUTION_SIZE_KB = 0.001953125 * metadata['vocab_size']
    MAX_CACHE_SIZE_KB = max_cache_gb * 1048576  # 1GB = 1048576KB
    MAX_CACHE_SIZE_TOKENS = MAX_CACHE_SIZE_KB / TOKEN_DISTRIBUTION_SIZE_KB

    mask = create_mask(metadata)

    def modify_distributions(model_output: torch.Tensor, conversation_tokenized: torch.Tensor) -> torch.Tensor:
        modified_output = F.softmax(model_output, dim=1)
        for i in range(conversation_tokenized.size(0) - 1):
            next_token = conversation_tokenized[i + 1]
            current_distribution = modified_output[i]
            max_prob = torch.max(current_distribution)
            if next_token == eos_token_id:
                current_distribution[eos_token_id] = max_prob * next_token_prob_boost
            else:
                if set_max_token_prob:
                    current_distribution[next_token] = max_prob * next_token_prob_boost
                elif next_token_prob_boost != 1.0:
                    current_distribution[next_token] *= next_token_prob_boost
            current_distribution /= current_distribution.sum()
        return modified_output


    def process_conversation(conversation_tokenized: torch.Tensor, conversation_content_ranges: list):
        conv_tokenized_gpu = conversation_tokenized.to(device)[:context_len]
        conv_distributions = preprocess_logits(model.forward(conv_tokenized_gpu.unsqueeze(0)).squeeze(0), mask)
        conv_distributions = modify_distributions(conv_distributions, conv_tokenized_gpu)
        conv_distributions_cpu = conv_distributions.to("cpu")
        return [conv_distributions_cpu[start:end] for start, end in conversation_content_ranges]

    dataset_distributions = []
    current_cache_size_tokens = 0
    count = 0
    total_saved_tokens = 0
    eos_token_id = metadata['eos_id']

    for conv_id, (conversation_tokenized, conversation_content_ranges) in enumerate(tqdm(zip(dataset_tokenized, dataset_content_ranges), desc="Generating Distributions", unit="convo", total=len(dataset_tokenized))):
        if conv_id in empty_convo_ids:
            dataset_distributions.append(torch.tensor([], device=device))
            continue

        content_distributions = process_conversation(conversation_tokenized, conversation_content_ranges)
        conversation_content_distributions = torch.cat(content_distributions, dim=0)

        dataset_distributions.append(conversation_content_distributions)

        total_saved_tokens += conversation_content_distributions.size(0)
        current_cache_size_tokens += conversation_content_distributions.size(0)

        if conversation_content_distributions.size(0) == 0:
            print(f"Empty convo id: {conv_id}")

        if current_cache_size_tokens >= MAX_CACHE_SIZE_TOKENS:
            count += 1
            async_save_partial_distributions(dataset_distributions, count)
            dataset_distributions = []
            current_cache_size_tokens = 0

    print(f"Total saved tokens: {total_saved_tokens}")
    if dataset_distributions:
        count += 1
        async_save_partial_distributions(dataset_distributions, count)


# Main Script
model_path = r"F:\UtopiaXL-13B"
dataset_path = r"F:\down\data-MNHTN-standardized-OpenPlatypus-Train.jsonl"
distributions_save_folder = r"F:\distilled"
context_len = 8*1024
#batch_size = 4 # How many conversations to process in parallel #TODO
chunk_size = 8 # How many `chunk_size_tokens` chunks to process in parallel
chunk_size_tokens = 1*1024
max_cache_gb = 10  # Desired size of the saved files in GB

set_max_token_prob = False # Set the probability of the next token to the max logit in the distribution
next_token_prob_boost = 1  # Boost the probability of the next token by this factor
sort = False # Sort by length, stops Vram spikes for some reason. Top - longest, Bottom - shortest

only_get_metadata = False # Won't load the model
test_inference = False

# Model settings
save_sys_range = False
save_user_range = False
save_assistant_range = True

prompt_format = {
    'SYS_START': "### System:\n",
    'USER_START': "### Instruction:\n",
    'ASSISTANT_START': "### Response:\n",
    'SYS_END': '\n\n',
    'USER_END': '\n\n',
    'ASSISTANT_END': '<eos>\n\n' # Use <eos> and <bos> for model-specific special tokens
}

device = "cuda:0"
model_name = os.path.basename(os.path.normpath(model_path))
dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
save_folder = os.path.join(distributions_save_folder, dataset_name, model_name)

print(f"Model: {model_name}\nDataset: {dataset_name}")

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

config_data = {
    'sorted': sort,
    'save_sys_range': save_sys_range,
    'save_user_range': save_user_range,
    'save_assistant_range': save_assistant_range,
    'context_len': context_len,
    'next_token_prob_boost': next_token_prob_boost,
    'set_max_token_prob': set_max_token_prob
}

if only_get_metadata:
    metadata = {**config_data, **generate_metadata(model_path, [], [])}
    save_sorted_dataset(save_folder, dataset_path)
    save_metadata(metadata, save_folder)
    print("Metadata saved!")
else:
    model = load_model(model_path, context_len, chunk_size, chunk_size_tokens)
    if test_inference:
        run_test_inference(model, prompt_format, model_path, device, text="Hey there, my name is")
        exit(0)
    if sort: save_sorted_dataset(save_folder, dataset_path)

    dataset_tokenized, dataset_content_ranges, empty_convo_ids = tokenize_dataset(dataset_path, device, sort, model_path, prompt_format, save_sys_range, save_user_range, save_assistant_range)
    metadata = {**config_data, "empty_convo_ids": empty_convo_ids, **generate_metadata(model_path, dataset_tokenized, dataset_content_ranges)}
    save_dataset_and_metadata(dataset_tokenized, dataset_content_ranges, metadata, save_folder)

    generate_probability_distributions(dataset_tokenized, dataset_content_ranges, model, device, max_cache_gb, next_token_prob_boost, context_len, metadata, empty_convo_ids)
    print("Done!\nIf the script didn't close yet, that means its still writing to disk!\nDO NOT STOP IT MANUALLY!")
