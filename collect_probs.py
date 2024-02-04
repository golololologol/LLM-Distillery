import os
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from tqdm import tqdm
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache
from utils.dataset_utils import H5Writer, save_dataset_and_metadata, \
    tokenize_dataset, generate_metadata, save_metadata, \
    good_encode, encode_prompt_format, save_sorted_dataset,\
    get_special_tokens, get_vocab_family
from utils.convert_to_safetensor import convert_model
@torch.inference_mode()

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

def is_model_safetensors(model_path: str):
    if os.path.isdir(model_path):
        for file in os.listdir(model_path):
            if file.endswith('.safetensors') and file.startswith('model'):
                return True
        return False
    
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


def generate_probability_distributions(dataset_tokenized, dataset_content_ranges, model, metadata, empty_convo_ids, clip_to_size):
    TOKEN_DISTRIBUTION_SIZE_KB = 0.001953125 * clip_to_size
    approx_size = round((metadata['total_content_tokens'] * TOKEN_DISTRIBUTION_SIZE_KB) / 1000000, 2)
    print(f"Approximate size of the final distributions: {approx_size}GB")

    def modify_distributions(model_output: torch.Tensor, conversation_tokenized: torch.Tensor) -> torch.Tensor:
        model_output = F.softmax(model_output, dim=1)
        for i in range(conversation_tokenized.size(0) - 1):
            next_token = conversation_tokenized[i + 1]
            current_distribution = model_output[i]
            if next_token == eos_token_id:
                current_distribution[eos_token_id] = torch.max(current_distribution) * next_token_prob_boost
            else:
                if set_max_token_prob:
                    current_distribution[next_token] = torch.max(current_distribution) * next_token_prob_boost
        
        model_output = model_output / model_output.sum(dim=-1, keepdim=True)
        return model_output

    def process_conversation(conversation_tokenized: torch.Tensor, conversation_content_ranges: list):
        conv_tokenized_gpu = conversation_tokenized.to(device)[:context_len]
        conv_distributions = model.forward(conv_tokenized_gpu.unsqueeze(0)).squeeze(0)[:, :clip_to_size]
        conv_distributions = modify_distributions(conv_distributions, conv_tokenized_gpu)
        conv_distributions_cpu = conv_distributions.to("cpu")
        return [conv_distributions_cpu[start:end] for start, end in conversation_content_ranges]

    total_saved_tokens = 0
    eos_token_id = metadata['eos_id']
    writer = H5Writer(save_folder)

    for conv_id, (conversation_tokenized, conversation_content_ranges) in enumerate(tqdm(zip(dataset_tokenized, dataset_content_ranges), desc="Generating Distributions", unit="convo", total=len(dataset_tokenized), smoothing=0.06)):
        if conv_id in empty_convo_ids:
            conversation_content_distributions = torch.tensor([], device=device)
            writer.write_data(conversation_content_distributions)
            continue

        content_distributions = process_conversation(conversation_tokenized, conversation_content_ranges)
        conversation_content_distributions = torch.cat(content_distributions, dim=0)

        total_saved_tokens += conversation_content_distributions.size(0)

        writer.write_data(conversation_content_distributions)
    writer.close()
    print(f"Total saved tokens: {total_saved_tokens}")


# Main Script
model_path = r"C:\Users\gololo\Desktop\text-generation-webui-main\models\DarkForest-20B-v1.0"
dataset_path = r"F:\down\vsakoye\randoBS.jsonl"
distributions_save_folder = r"F:\distilled"
context_len = 2*1024
#batch_size = 4 # How many conversations to process in parallel #TODO
chunk_size = 4 # How many `chunk_size_tokens` chunks to process in parallel
chunk_size_tokens = 2*1024

set_max_token_prob = False # Set the probability of the next token to the max logit in the distribution
next_token_prob_boost = 1.15 # Boost the probability of the next token by this factor
sort = False # Sort by length, stops Vram spikes for some reason. Top - longest, Bottom - shortest

only_get_metadata = False # Won't load the model
test_inference = False

# Model settings
save_sys_range = False
save_user_range = False
save_assistant_range = True
crop_distr_to_size = 32000
device = "cuda:1"

prompt_format = {
    'SYS_START': "<im_start>system\n",
    'USER_START': "<im_start>user\n",
    'ASSISTANT_START': "<im_start>assistant\n",
    'SYS_END': '\n<im_end>\n',
    'USER_END': '\n<im_end>\n',
    'ASSISTANT_END': '\n<im_end>\n' # Use <eos> and <bos> for model-specific special tokens
}

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
    'crop_to_size': crop_distr_to_size,
    'next_token_prob_boost': next_token_prob_boost if set_max_token_prob else 1,
    'set_max_token_prob': set_max_token_prob
}

if only_get_metadata:
    metadata = {**config_data, **generate_metadata(model_path, [], [])}
    save_sorted_dataset(save_folder, dataset_path)
    save_metadata(metadata, save_folder)
    print("Metadata saved!")
else:
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

    if sort: save_sorted_dataset(save_folder, dataset_path)

    dataset_tokenized, dataset_content_ranges, empty_convo_ids = tokenize_dataset(dataset_path, device, sort, model_path, prompt_format, context_len, save_sys_range, save_user_range, save_assistant_range)
    metadata = {**config_data, **generate_metadata(model_path, dataset_tokenized, dataset_content_ranges, empty_convo_ids=empty_convo_ids)}
    save_dataset_and_metadata(dataset_tokenized, dataset_content_ranges, metadata, save_folder)

    generate_probability_distributions(dataset_tokenized, dataset_content_ranges, model, metadata, empty_convo_ids, crop_distr_to_size)
    print("Done!")
