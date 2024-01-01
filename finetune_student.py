import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, get_scheduler
import pickle
import os
from tqdm import tqdm
from utils.tokenize_dataset import tokenize_dataset
from utils.dataset_utils import save_dataset_and_metadata
from utils.dataset_utils import load_metadata

def check_errors(dataset_path: str, distributions_path: str, student_metadata: dict, distr_metadata: dict):
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    distributions_dataset_name = distributions_path.split(os.sep)[-2] if len(distributions_path.split(os.sep)) > 1 else None
    kill_it = False
    if dataset_name != distributions_dataset_name:
        print("DIFFERENT DATASET NAMES!")
        kill_it = True
    if (student_metadata['vocab_family'] != distr_metadata['vocab_family']) and (student_metadata['vocab_family'] != "Unknown") and (distr_metadata['vocab_family'] != "Unknown"):
        print("DIFFERENT VOCAB FAMILIES!")
        kill_it = True
    if student_metadata['vocab_size'] != distr_metadata['vocab_size']:
        print("DIFFERENT VOCAB SIZES!")
        kill_it = True
    if student_metadata['vocab_family'] == "Unknown":
        print("UNKNOWN STUDENT VOCAB FAMILY!")
        kill_it = True
    if distr_metadata['vocab_family'] == "Unknown":
        print("UNKNOWN TEACHER VOCAB FAMILY!")
        kill_it = True
    if kill_it:
        raise Exception("ERRORS FOUND! CHECK ABOVE FOR DETAILS!")

def teacher_tensors_hander(distributions_path, device):
    while True:
        files = sorted([f for f in os.listdir(distributions_path) if f.startswith("distributions_") and f.endswith(".pkl")])
        for file in files:
            file_path = os.path.join(distributions_path, file)
            with open(file_path, 'rb') as f:
                numpy_tensor_list = pickle.load(f)
                for numpy_tensor in numpy_tensor_list:
                    yield torch.tensor(numpy_tensor, device=device)

def calculate_kl_divergence(student_logits, teacher_logits):
    student_log_probs = nn.functional.log_softmax(student_logits, dim=-1)
    teacher_probs = nn.functional.softmax(teacher_logits, dim=-1)
    kl_div = nn.functional.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
    return kl_div

def finetune(model_path: str, save_folder: str, dataset_tokenized: list, dataset_content_ranges: list, distr_context_len: int, distributions_path: str, context_length: int, num_epochs: int, num_warmup_steps: int, lr: float, device: str):
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map="auto")
    model.train()
    teacher_tensor = teacher_tensors_hander(distributions_path, device)
    dataset_len = len(dataset_tokenized)
    crop_to_len = min(distr_context_len, context_length)
    num_training_steps = num_epochs * len(dataset_tokenized)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)
    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_training_steps)
    recent_losses = []
    pbar = tqdm(total=num_training_steps, desc="Finetuning", unit="convo", smoothing=0.06)

    for step in range(num_training_steps):
        conversation_tokenized, conversation_content_ranges = dataset_tokenized[step % dataset_len], dataset_content_ranges[step % dataset_len]
        conversation_tokenized = conversation_tokenized.to(device).unsqueeze(0)
        full_student_logits = model(conversation_tokenized).logits.squeeze(0)
        student_logits = torch.cat([full_student_logits[start:end] for start, end in conversation_content_ranges], dim=0)[:crop_to_len]

        teacher_logits = next(teacher_tensor)[:crop_to_len]

        loss = calculate_kl_divergence(student_logits, teacher_logits)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        recent_losses.append(loss.item())
        if len(recent_losses) > dataset_len:
            recent_losses.pop(0)

        # Calculate average of recent losses
        avg_recent_loss = sum(recent_losses) / len(recent_losses)

        # Update progress bar
        pbar.set_postfix({"Epoch": step / dataset_len, "Avg Dataset Loss": avg_recent_loss})
        pbar.update()
    pbar.close()
    model.save_pretrained(save_folder)
    print(f"Model successfully saved at {save_folder}")



# Main Script
model_path = r"C:\Users\gololo\Desktop\TinyLlama-1.1B-intermediate-step-1431k-3T"
dataset_path = r"F:\down\theoremqa.jsonl"
#validation_dataset_path = r"C:\Users\gololo\Documents\janny\janny_Filteredtest.jsonl"
distributions_path = r"F:\distilled\theoremqa\speechless-llama2-hermes-orca-platypus-wizardlm-13b-exl2"
save_folder = r"F:\trained"
trained_model_name = r"BallCrusher9000"
context_length = 2*1024
num_epochs = 5
num_warmup_steps = 200
lr = 3e-5

prompt_format = {
    'SYS_START': "### System:\n",
    'USER_START': "### User:\n",
    'ASSISTANT_START': "### Assistant:\n",
    'SYS_END': '\n\n',
    'USER_END': '\n\n',
    'ASSISTANT_END': '<eos>\n\n' # Use <eos> and <bos> for model-specific special tokens
}

device = "cuda:0"

trained_model_folder = os.path.join(save_folder, trained_model_name)

if not os.path.exists(trained_model_folder):
    os.makedirs(trained_model_folder)

distr_metadata = load_metadata(distributions_path)

if distr_metadata is not None:
    dataset_tokenized, dataset_content_ranges, student_metadata = tokenize_dataset(
        dataset_path, device, distr_metadata['sorted'], model_path, prompt_format, context_length, 
        distr_metadata['save_sys_range'], distr_metadata['save_user_range'], 
        distr_metadata['save_assistant_range'])
    
    save_dataset_and_metadata(dataset_tokenized, dataset_content_ranges, student_metadata, trained_model_folder)
    
    check_errors(dataset_path, distributions_path, student_metadata, distr_metadata)

    finetune(model_path, trained_model_folder, dataset_tokenized, dataset_content_ranges, distr_metadata['context_len'], distributions_path, context_length, num_epochs, num_warmup_steps, lr, device)