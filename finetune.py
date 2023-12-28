import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import os
from tqdm import tqdm
import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_tokenized_dataset(dataset_tokenized: list, dataset_content_ranges: list):
    file = save_folder + "/dataset_tokenized.jsonl"
    with open(file, 'w', encoding='utf-8') as f:
        for i, convo_tokenized in enumerate(dataset_tokenized):
            content_tokens = []
            for content_range in dataset_content_ranges[i]:
                content_start, content_end = content_range
                content_tokens.extend(convo_tokenized[content_start:content_end].tolist())

            data_to_save = {
                "convo_tokenized": convo_tokenized.tolist(),
                "content_ranges": dataset_content_ranges[i],
                "Content_tokens": content_tokens
            }

            f.write(json.dumps(data_to_save, ensure_ascii=False) + '\n')

def tokenize_dataset(dataset: list):
    print("Tokenizing the dataset")
    total_tokens = 0
    
    def good_encode(text: str, encode_special = True, replace_tokens = False):
        if replace_tokens:
            return tokenizer.encode(f"\n{text.replace('<bos>', tokenizer.bos_token).replace('<eos>', tokenizer.eos_token)}", encode_special_tokens=encode_special).squeeze(0)[2:] # type: ignore
        return tokenizer.encode(f"\n{text}", encode_special_tokens=encode_special).squeeze(0)[2:] # type: ignore
    
    dataset_tokenized = []
    dataset_content_ranges = []
    user_start_tokenized = good_encode(USER_START, replace_tokens=True)
    user_end_tokenized = good_encode(USER_END, replace_tokens=True)
    assistant_start_tokenized = good_encode(ASSISTANT_START, replace_tokens=True)
    assistant_end_tokenized = good_encode(ASSISTANT_END, replace_tokens=True)
    content_start_offset = assistant_start_tokenized.numel() - 1
    content_end_offset = assistant_end_tokenized.numel()

    for item in dataset: # Every conversation
        conversation_tokenized = torch.Tensor().to(device)
        conversation_content_ranges = []
        start_index = 0

        sys_content = item["init"]
        if sys_content:
            sys_finalized = SYS_START + sys_content.strip() + SYS_END
            sys_tokenized = good_encode(sys_finalized, replace_tokens=True)
            conversation_tokenized = sys_tokenized
            start_index = sys_tokenized.numel()

        for i, turn in enumerate(item["conversations"]): # Every turn
            assistant = i % 2

            turn_start_tokenized = assistant_start_tokenized if assistant else user_start_tokenized
            turn_end_tokenized = assistant_end_tokenized if assistant else user_end_tokenized
            turn_tokenized = good_encode(turn.strip(), encode_special=False)

            full_turn_tokenized = torch.cat((turn_start_tokenized, turn_tokenized, turn_end_tokenized))
            end_index = start_index + full_turn_tokenized.numel()

            if assistant:
                content_start_index = start_index + content_start_offset
                content_end_index = end_index - content_end_offset
                conversation_content_ranges.append((content_start_index, content_end_index))

            start_index = end_index

            conversation_tokenized = torch.cat((conversation_tokenized, full_turn_tokenized)) if conversation_tokenized.numel() > 0 else full_turn_tokenized
        total_tokens += conversation_tokenized.numel()
        dataset_tokenized.append(conversation_tokenized.to("cpu"))
        dataset_content_ranges.append(conversation_content_ranges)

    if sort:
        combined_list = list(zip(dataset_tokenized, dataset_content_ranges))
        combined_list.sort(key=lambda x: x[0].shape[0], reverse=True)
        dataset_tokenized, dataset_content_ranges = map(list, zip(*combined_list))

    print(f"Total processed tokens: {total_tokens}")
    return dataset_tokenized, dataset_content_ranges

def calculate_kl_divergence(student_logits, teacher_logits):
    student_probs = torch.nn.functional.softmax(student_logits, dim=-1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)
    kl_div = torch.nn.functional.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')
    return kl_div

def finetune(model, dataset_tokenized, dataset_content_ranges):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr)

    for epoch in range(num_epochs):
        total_loss = 0
        for conversation_tokenized, content_ranges in tqdm(zip(dataset_tokenized, dataset_content_ranges), desc="Finetuning", unit="convo", postfix=f"Epoch: {epoch}"):


        avg_loss = total_loss / len(dataset_tokenized)
        print(f"Epoch {epoch+1}: Average Loss = {avg_loss}")

# Main Script
model_path = r"C:\Users\gololo\Desktop\neural-chat-7b-v3-1-exl2"
dataset_path = r"C:\Users\gololo\Documents\janny\janny_Filtered.jsonl"
distributions_path = r"F:\distilled\janny_Filtered\neural-chat-7b-v3-1-exl2"
save_folder = r"F:\trained"
trained_model_name = r"BallCrusher9000"
max_input_len = 8192
window_size = 7168
num_epochs = 5
lr = 5e-5
sort = True # Sort by length, stops Vram spikes for some reason. Top - longest, Bottom - shortest

SYS_START = "### System\n"
USER_START = "### User\n"
ASSISTANT_START = "### Assistant\n"
SYS_END = "\n"
USER_END = "\n"
ASSISTANT_END = "<eos>\n" # Use <eos> and <bos> for model-specific special tokens

device = "cuda:0"
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)

dataset = read_jsonl(dataset_path)

dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
distributions_dataset_name = distributions_path.split(os.sep)[-2] if len(distributions_path.split(os.sep)) > 1 else None
if dataset_name != distributions_dataset_name:
    print("NUH UH, YOU JUST TRIED TO DISTILL FROM DATASETS WITH DIFFERENT NAMES!")
    os._exit
trained_model_save_folder = os.path.join(save_folder, trained_model_name)

if not os.path.exists(trained_model_save_folder):
    os.makedirs(trained_model_save_folder)

dataset_tokenized, dataset_content_ranges = tokenize_dataset(dataset)
save_tokenized_dataset(dataset_tokenized, dataset_content_ranges)
finetune(model, dataset_tokenized, dataset_content_ranges)