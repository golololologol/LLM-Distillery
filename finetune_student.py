import torch
import torch.nn as nn
from accelerate import Accelerator
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
import pickle
import os
from tqdm import tqdm
from utils.tokenize_dataset import tokenize_dataset
from utils.dataset_utils import save_dataset_and_metadata
from utils.dataset_utils import load_metadata

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
                numpy_tensor_list = pickle.load(f)
                for numpy_tensor in numpy_tensor_list:
                    yield torch.tensor(numpy_tensor)

def calculate_kl_divergence(student_logits, teacher_logits):
    student_probs = torch.nn.functional.softmax(student_logits, dim=-1)
    teacher_probs = torch.nn.functional.softmax(teacher_logits, dim=-1)
    kl_div = torch.nn.functional.kl_div(student_probs.log(), teacher_probs, reduction='batchmean')
    return kl_div

def finetune(model: nn.Module, dataset_tokenized: list, dataset_content_ranges: list, validation_dataset_tokenized:list, validation_dataset_ranges: list, distr_context_len: int, teacher_metadata: dict, student_metadata: dict, device: str):
    model.train()
    teacher_tensor = teacher_tensors_hander(distributions_path)
    dataset_len = len(dataset_tokenized)
    crop_to_len = min(distr_context_len, context_length)
    step = 0

    for epoch in range(num_epochs):
        total_loss = 0
        for conversation_tokenized, conversation_content_ranges in tqdm(zip(dataset_tokenized, dataset_content_ranges), total=dataset_len, desc="Finetuning", unit="convo", postfix={"Epoch": epoch, "Loss": total_loss/dataset_len}):
            conversation_tokenized = conversation_tokenized.to(device).unsqueeze(0)[:crop_to_len]
            teacher_logits = next(teacher_tensor).to(device)[:crop_to_len]

            full_student_logits = 
            student_logits = torch.cat([full_student_logits[start:end] for start, end in conversation_content_ranges], dim=0)

    # Save the fine-tuned model


# Main Script
model_path = r"C:\Users\gololo\Desktop\TinyLlama-1.1B-intermediate-step-1431k-3T"
dataset_path = r"C:\Users\gololo\Documents\janny\janny_Filteredtest.jsonl"
validation_dataset_path = r"C:\Users\gololo\Documents\janny\janny_Filteredtest.jsonl"
distributions_path = r"F:\distilled\janny_Filtered\neural-chat-7b-v3-1-exl2"
save_folder = r"F:\trained"
trained_model_name = r"BallCrusher9000"
context_length = 8192
num_epochs = 5
lr = 1e-4

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
model = AutoModelForCausalLM.from_pretrained(model_path)
model = DataParallel(model, device_ids=[0, 1], output_device=0)
model.to(device)

trained_model_save_folder = os.path.join(save_folder, trained_model_name)

if not os.path.exists(trained_model_save_folder):
    os.makedirs(trained_model_save_folder)

distr_metadata = load_metadata(distributions_path)

if distr_metadata is not None:
    dataset_tokenized, dataset_content_ranges, student_metadata = tokenize_dataset(
        dataset_path, device, distr_metadata['sorted'], model_path, prompt_format, context_length, 
        distr_metadata['save_sys_range'], distr_metadata['save_user_range'], 
        distr_metadata['save_assistant_range'])
    
    validation_dataset_tokenized, validation_dataset_ranges, _ = tokenize_dataset(
        validation_dataset_path, device, distr_metadata['sorted'], model_path, prompt_format, context_length, 
        distr_metadata['save_sys_range'], distr_metadata['save_user_range'], 
        distr_metadata['save_assistant_range'])
    
    save_dataset_and_metadata(dataset_tokenized, dataset_content_ranges, student_metadata, trained_model_save_folder)

    distr_context_len = distr_metadata['context_len']

    finetune(model, dataset_tokenized, dataset_content_ranges, validation_dataset_tokenized,
                  validation_dataset_ranges, distr_context_len, distr_metadata, student_metadata, device)