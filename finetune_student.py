import os
from finetuning.full_finetune import full_finetune
from finetuning.Lora import lora_finetune
from finetuning.QLora import qlora_finetune
from utils.dataset_utils import save_dataset_and_metadata, load_metadata, tokenize_dataset, generate_metadata, filter_empty_conversations

def check_errors(dataset_path: str, distributions_path: str, student_metadata: dict, distr_metadata: dict):
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    distributions_dataset_name = distributions_path.split(os.sep)[-2] if len(distributions_path.split(os.sep)) > 1 else None

    errors = []
    if dataset_name != distributions_dataset_name:
        errors.append("DIFFERENT DATASET NAMES!")
    if (student_metadata['vocab_family'] != distr_metadata['vocab_family']) and (student_metadata['vocab_family'] != "Unknown") and (distr_metadata['vocab_family'] != "Unknown"):
        errors.append("DIFFERENT VOCAB FAMILIES!")
    if student_metadata['vocab_size'] != distr_metadata['vocab_size']:
        errors.append("DIFFERENT VOCAB SIZES!")
    if student_metadata['vocab_family'] == "Unknown":
        errors.append("UNKNOWN STUDENT VOCAB FAMILY!")
    if distr_metadata['vocab_family'] == "Unknown":
        errors.append("UNKNOWN TEACHER VOCAB FAMILY!")

    if errors:
        for error in errors:
            print(error)
        print("----------------------------------------")
        raise Exception("Errors detected! Aborting!")


def finetune(parameters: dict):
    if training.lower() == "full":
        full_finetune(parameters)
    elif training.lower() == "lora":
        lora_finetune(parameters)
    elif training.lower() == "qlora":
        qlora_finetune(parameters)
    else:
        raise Exception("Invalid training type!")

# Main Script
model_path = r"C:\Users\gololo\Desktop\TinyLlama-1.1B-intermediate-step-1431k-3T"
dataset_path = r"C:\Users\gololo\Documents\janny\janny_Filtered.jsonl"
#validation_dataset_path = r"C:\Users\gololo\Documents\janny\janny_Filteredtest.jsonl"
distributions_path = r"F:\distilled\janny_Filtered\speechless-llama2-hermes-orca-platypus-wizardlm-13b-exl2"
save_folder = r"F:\trained"
trained_model_name = "BallCrusher9000"

training = "full" # "full" "lora" "qlora"
optimizer = "adamw8bit" # "adamw8bit" "adamw" "adagrad8bit" "sgd" "paged_adamw8bit"
load_in_8bit = False
context_length = 2*1024
grad_accumulation_steps = 4
num_epochs = 4
num_warmup_steps = 120
lr = 1e-6
lr_scheduler = "linear"

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

print(f"Context Len: {context_length}, Num Epochs: {num_epochs}, Num Warmup Steps: {num_warmup_steps}, LR: {lr} {lr_scheduler}, Optimizer: {optimizer}, Prob Boost: {distr_metadata['next_token_prob_boost']}, Set Prob to Max: {distr_metadata['set_max_token_prob']}")

if distr_metadata is not None:
    dataset_tokenized, dataset_content_ranges, empty_convo_ids = tokenize_dataset(
        dataset_path, device, distr_metadata['sorted'], model_path, prompt_format, 
        distr_metadata['save_sys_range'], distr_metadata['save_user_range'], 
        distr_metadata['save_assistant_range'])
    student_metadata = generate_metadata(model_path, dataset_tokenized, dataset_content_ranges)
    
    empty_convo_ids = list(set(distr_metadata.get('empty_convo_ids', []) + empty_convo_ids))

    filter_empty_conversations(dataset_tokenized, dataset_content_ranges, empty_convo_ids)

    save_dataset_and_metadata(dataset_tokenized, dataset_content_ranges, student_metadata, trained_model_folder)
    
    check_errors(dataset_path, distributions_path, student_metadata, distr_metadata)

    parameters = {
        "model_path": model_path,
        "save_folder": trained_model_folder,
        "dataset_tokenized": dataset_tokenized,
        "dataset_content_ranges": dataset_content_ranges,
        "distributions_path": distributions_path,
        "empty_convo_ids": empty_convo_ids,
        "training_type": training,
        "load_in_8bit": load_in_8bit,
        "optimizer_name": optimizer,
        "context_length": context_length,
        "grad_accum_steps": grad_accumulation_steps,
        "num_epochs": num_epochs,
        "num_warmup_steps": num_warmup_steps,
        "student_metadata": student_metadata,
        "lr": lr,
        "lr_scheduler_name": lr_scheduler,
        "device": device,
    }

    finetune(parameters)