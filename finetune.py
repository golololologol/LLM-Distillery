import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
import pickle
import os
from tqdm import tqdm
from tokenize_dataset import tokenize_dataset
import json

def read_jsonl(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            data.append(json.loads(line))
    return data

def save_tokenized_dataset(dataset_tokenized: list, dataset_content_ranges: list, save_folder: str):
    file = os.path.join(save_folder, "dataset_tokenized.jsonl")
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

def load_metadata(distributions_path):
    metadata_path = distributions_path / "dataset_metadata.jsonl"
    try:
        with open(metadata_path, 'r') as file:
            for line in file:
                return json.loads(line)
    except FileNotFoundError:
        print(f"Metadata file not found at {metadata_path}")
        os._exit

def check_dataset_names(dataset_path: str, distributions_path: str):
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    distributions_dataset_name = distributions_path.split(os.sep)[-2] if len(distributions_path.split(os.sep)) > 1 else None
    if dataset_name != distributions_dataset_name:
        print("NUH UH, YOU JUST TRIED TO DISTILL FROM DATASETS WITH DIFFERENT NAMES!")
        os._exit

def teacher_tensors_hander(distributions_path):
    while True:
        files = sorted([f for f in os.listdir(distributions_path) if f.startswith("distributions_") and f.endswith(".pkl")])
        for file in files:
            file_path = os.path.join(distributions_path, file)
            with open(file_path, 'rb') as f:
                tensor_list = pickle.load(f)
                for tensor in tensor_list:
                    yield tensor

def ensure_compatibility(teacher_logits, index_map):
    compatible_teacher_distr = torch.empty_like(teacher_logits[0])

    for teacher_distr in teacher_logits:
        for idx2, idx1 in index_map.items():
            compatible_teacher_distr[idx1] = teacher_distr[idx2]

    return compatible_teacher_distr

def calculate_kl_divergence(student_logits, teacher_logits):
    student_probs = torch.nn.functional.softmax(student_logits, dim=-1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)
    kl_div = torch.nn.functional.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')
    return kl_div

def finetune(model: nn.Module, dataset_tokenized: list, dataset_content_ranges: list, distr_context_len: int, teacher_metadata: dict, student_metadata: dict):
    teacher_dict = teacher_metadata['vocab']
    student_dict = student_metadata['vocab']
    all_tokens_present = all(token in student_dict for token in teacher_dict)
    if all_tokens_present:
        index_map = {teacher_dict[token]: student_dict[token] for token in teacher_dict}
        sorted_indices = [index_map[i] for i in range(len(index_map))]
    else:
        index_map = {}
        sorted_indices = []
        print("Teacher and student vocabularies are not compatible!")
        os._exit

    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr)
    teacher_tensor = teacher_tensors_hander(distributions_path)
    dataset_len = len(dataset_tokenized)
    crop_to_len = min(distr_context_len, context_length)
    step = 0

    for epoch in range(num_epochs):
        total_loss = 0
        for conversation_tokenized, conversation_content_ranges in tqdm(zip(dataset_tokenized, dataset_content_ranges), total=dataset_len, desc="Finetuning", unit="convo", postfix={"Epoch": epoch, "Loss": total_loss/dataset_len}):
            conversation_tokenized = torch.tensor(conversation_tokenized).to(device)
            teacher_logits = next(teacher_tensor).to(device)[:crop_to_len]

            # Generating logits from the student model
            full_student_logits = model(conversation_tokenized, labels=conversation_tokenized)[1].squeeze()[:crop_to_len]
            student_logits = torch.cat([full_student_logits[start:end] for start, end in conversation_content_ranges], dim=0)
            
            # Ensure compatibility between teacher and student logits
            teacher_logits = teacher_logits[:, sorted_indices]

            # Calculate loss
            loss = calculate_kl_divergence(student_logits, teacher_logits)
            total_loss += loss.item()

            # Backpropagation
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            step += 1

    # Save the fine-tuned model
    model.save_pretrained(trained_model_save_folder)


# Main Script
model_path = r"C:\Users\gololo\Desktop\neural-chat-7b-v3-1-exl2"
dataset_path = r"C:\Users\gololo\Documents\janny\janny_Filtered.jsonl"
distributions_path = r"F:\distilled\janny_Filtered\neural-chat-7b-v3-1-exl2"
save_folder = r"F:\trained"
trained_model_name = r"BallCrusher9000"
context_length = 8192
num_epochs = 5
lr = 5e-5

prompt_format = {
    'SYS_START': "### System:\n",
    'USER_START': "### User:\n",
    'ASSISTANT_START': "### Assistant:\n",
    'SYS_END': '\n',
    'USER_END': '\n',
    'ASSISTANT_END': '<eos>\n' # Use <eos> and <bos> for model-specific special tokens
}

device = "cuda:0"

check_dataset_names(dataset_path, distributions_path)
model = AutoModelForCausalLM.from_pretrained(model_path).to(device)
tokenizer = AutoTokenizer.from_pretrained(model_path)
dataset = read_jsonl(dataset_path)
trained_model_save_folder = os.path.join(save_folder, trained_model_name)

if not os.path.exists(trained_model_save_folder):
    os.makedirs(trained_model_save_folder)

distr_metadata = load_metadata(distributions_path)

if distr_metadata is not None:
    dataset_tokenized, dataset_content_ranges, student_metadata = tokenize_dataset(
        dataset, device, distr_metadata['sort'], model_path, True, prompt_format, 999, 
        distr_metadata['save_sys_range'], distr_metadata['save_user_range'], 
        distr_metadata['save_assistant_range'])
    save_tokenized_dataset(dataset_tokenized, dataset_content_ranges, trained_model_save_folder)
    distr_context_len = distr_metadata['context_len']
    if student_metadata is not None:
        finetune(model, dataset_tokenized, dataset_content_ranges, distr_context_len, distr_metadata, student_metadata)