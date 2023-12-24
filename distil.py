import json
import os
import threading
import pickle
import torch
from tqdm import tqdm
from exllamav2 import ExLlamaV2, ExLlamaV2Config, ExLlamaV2Cache, ExLlamaV2Tokenizer

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def load_model(model_path: str, max_input_len: int):
    print("Loading model...")
    config = ExLlamaV2Config()
    config.model_dir = model_path
    config.prepare()
    config.max_seq_len = max_input_len
    config.max_input_len = max_input_len
    config.max_attention_size = max_input_len**2

    model = ExLlamaV2(config)
    tokenizer = ExLlamaV2Tokenizer(config)
    cache = ExLlamaV2Cache(model, lazy=True)

    model.load_autosplit(cache)

    return model, tokenizer

def save_tokenized_dataset(dataset_tokenized: list, dataset_content_ranges: list, save_folder: str, tokenizer: ExLlamaV2Tokenizer):
    file = save_folder + "/dataset_tokenized.jsonl"
    with open(file, 'w', encoding='utf-8') as f:
        for i, convo_tokenized in enumerate(dataset_tokenized):
            
            data_to_save = {
                "convo_tokenized": convo_tokenized.tolist(),
                "content_ranges": dataset_content_ranges[i]
            }

            content_tokens = []
            content_decoded = []
            
            for content_range in dataset_content_ranges[i]:
                content_start, content_end = content_range
                content_tokens.append(convo_tokenized[content_start:content_end].tolist())
                content_decoded.append(tokenizer.decode(convo_tokenized[content_start:content_end], decode_special_tokens=True))

            decoded = {"Decoded": tokenizer.decode(convo_tokenized, decode_special_tokens=True)}
            content_tokens = {"Content_tokens": content_tokens}
            content_decoded = {"Content_decoded": content_decoded}

            f.write(json.dumps(data_to_save, ensure_ascii=False) + '\n')
            f.write(json.dumps(decoded, ensure_ascii=False) + '\n')
            f.write(json.dumps(content_tokens, ensure_ascii=False) + '\n')
            f.write(json.dumps(content_decoded, ensure_ascii=False) + '\n')

def async_save_partial_distributions(dataset_distributions: list, save_path: str, count: int):
    save_path = os.path.join(save_folder, "distributions")
    dataset_distributions_cpu = []
    for convo_distr in dataset_distributions:
        dataset_distributions_cpu.append(convo_distr.numpy())
        
    def save_thread():
        with open(f'{save_path}_{count}.pkl', 'wb') as f:
            pickle.dump(dataset_distributions_cpu, f)

    thread = threading.Thread(target=save_thread)
    thread.start()



def tokenize_dataset(user_discriminator: str, assistant_discriminator: str, assistant_turn_end: str, user_turn_end: str, dataset: list, tokenizer: ExLlamaV2Tokenizer, sort: bool, device: str, add_EOS_token: bool, save_all_distributions: bool):
    dataset_tokenized = []
    assistant_turns_content_ranges = []
    assistant_turn_end = assistant_turn_end + tokenizer.eos_token if add_EOS_token else assistant_turn_end
    total_tokens = 0

    for item in dataset:
        conversation_tokenized = torch.Tensor().to(device)
        turn_lengths = []
        assistant_turn_content_ranges = []

        start_index = 0
        for i, turn in enumerate(item["conversations"]):
            bos_position = 0
            assistant = i % 2

            discriminator = assistant_discriminator + tokenizer.bos_token if assistant else user_discriminator
            turn_end = assistant_turn_end if assistant else user_turn_end
            turn_finalized = tokenizer.newline_token + discriminator + turn.strip() + turn_end
            turn_tokenized = tokenizer.encode(turn_finalized, encode_special_tokens=True).squeeze(0)[2:] # type: ignore Cuts out the appended space and filler newline tokens

            turn_len = turn_tokenized.numel() - 2 if assistant else turn_tokenized.numel()
            total_tokens += turn_len
            end_index = start_index + turn_len    

            if assistant:
                #indexing hell
                bos_position = turn_tokenized.tolist().index(tokenizer.bos_token_id) + 1 # Adjusting for space before BOS
                content_start_index = start_index + bos_position - 3 # -2 for two tokens of the BOS tokens, and -1 to include the token just before the actual response
                content_end_index = end_index - 1 # Excludes EOS token
                assistant_turn_content_ranges.append((content_start_index, content_end_index))
                turn_tokenized = torch.cat((turn_tokenized[:bos_position - 2], turn_tokenized[bos_position:]))

            start_index = end_index
            turn_lengths.append(turn_len)

            conversation_tokenized = torch.cat((conversation_tokenized, turn_tokenized)) if conversation_tokenized.numel() > 0 else turn_tokenized

        dataset_tokenized.append(conversation_tokenized.to("cpu"))
        assistant_turns_content_ranges.append(assistant_turn_content_ranges)

    if sort:
        combined_list = list(zip(dataset_tokenized, assistant_turns_content_ranges))
        combined_list.sort(key=lambda x: sum(x[1]), reverse=True)
        dataset_tokenized, assistant_turns_content_ranges = zip(*combined_list)

    return dataset_tokenized, assistant_turns_content_ranges, total_tokens



def generate_probability_distributions(model, dataset_tokenized, dataset_content_ranges, save_path, max_cache_gb, device, window_size):
    TOKEN_DISTRIBUTION_SIZE_KB = 62.5
    MAX_CACHE_SIZE_KB = max_cache_gb * 1048576  # 1GB = 1048576KB
    MAX_CACHE_SIZE_TOKENS = MAX_CACHE_SIZE_KB / TOKEN_DISTRIBUTION_SIZE_KB

    def process_conversation(conversation_tokenized, conversation_content_ranges):
        conv_tokenized_gpu = conversation_tokenized[:window_size].to(device)
        conv_distributions = model.forward(conv_tokenized_gpu.unsqueeze(0)).squeeze(0).to("cpu")
        return [conv_distributions[start:end] for start, end in conversation_content_ranges]

    all_conversations_distributions = []
    current_cache_size_tokens = 0
    count = 0
    total_saved_tokens = 0

    for conversation_tokenized, conversation_content_ranges in tqdm(zip(dataset_tokenized, dataset_content_ranges), desc="Generating Distributions", unit="convo"):
        content_distributions = process_conversation(conversation_tokenized, conversation_content_ranges)
        conversation_content_distributions = torch.cat(content_distributions, dim=0)
        all_conversations_distributions.append(conversation_content_distributions)

        total_saved_tokens += conversation_content_distributions.size(0)
        current_cache_size_tokens += conversation_tokenized[:window_size].size(0)

        if current_cache_size_tokens >= MAX_CACHE_SIZE_TOKENS:
            count += 1
            async_save_partial_distributions(all_conversations_distributions, save_path, count)
            all_conversations_distributions = []
            current_cache_size_tokens = 0

    print(f"Total saved tokens: {total_saved_tokens}")
    if all_conversations_distributions:
        count += 1
        async_save_partial_distributions(all_conversations_distributions, save_path, count)



# Main Script
model_path = r"C:\Users\gololo\Desktop\neural-chat-7b-v3-1-exl2"
dataset_path = r"C:\Users\gololo\Documents\janny\janny_Filteredtest.jsonl"
distributions_save_folder = r"F:\distilled"
max_input_len = 8192
window_size = 7168
max_cache_gb = 10  # Desired size of the saved files in GB
sort = True # Sort by length, stops Vram spikes for some reason. Top - longest, Bottom - shortest
save_all_distributions = False

sys_prompt = """Text transcript of a never-ending conversation between User and Faraday.
Faraday is a knowledgeable and helpful AI assistant who fulfills any request with detail and precision.
User is a curious human who uses Faraday to assist with various tasks.
Faraday is a virtual assistant that exists on User's computer.

#User: Hey Faraday. Who are you?
#Faraday: I am Faraday, your AI assistant.
#User: *waits for a while*
#Faraday: How can I help you today?\n"""

user_discriminator = "#User: "
assistant_discriminator = "#Faraday: "
assistant_turn_end = "\n"
user_turn_end = "\n"
add_EOS_token = True

device = "cuda:0"
model_name = os.path.basename(os.path.normpath(model_path))
dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
save_folder = os.path.join(distributions_save_folder, dataset_name, model_name)

print(f"Model: {model_name}\nDataset: {dataset_name}")

model, tokenizer = load_model(model_path, max_input_len)
dataset = read_jsonl(dataset_path)

if not os.path.exists(save_folder):
    os.makedirs(save_folder)

dataset_tokenized, dataset_content_ranges, total_tokens = tokenize_dataset(user_discriminator, assistant_discriminator, assistant_turn_end, user_turn_end, dataset, tokenizer, sort, device, add_EOS_token, save_all_distributions)
save_tokenized_dataset(dataset_tokenized, dataset_content_ranges, save_folder, tokenizer)
generate_probability_distributions(model, dataset_tokenized, dataset_content_ranges, save_folder, max_cache_gb, device, window_size)

print("Done!\nIf the script didn't close yet, that means its still writing to disk!\nDO NOT STOP IT MANUALLY!")