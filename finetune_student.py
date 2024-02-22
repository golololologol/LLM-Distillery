import os

from numpy import add
from regex import F
from finetuning.full_finetune import full_finetune
from finetuning.Lora import lora_finetune
from finetuning.QLora import qlora_finetune
from utils.dataset_utils import save_dataset_and_metadata, load_metadata, tokenize_dataset, generate_metadata, filter_empty_conversations
from utils.finetuning_utils import Params

def check_errors(dataset_path: str, distributions_path: str, student_metadata: dict, distr_metadata: dict):
    dataset_name = os.path.splitext(os.path.basename(dataset_path))[0]
    distributions_dataset_name = distributions_path.split(os.sep)[-2] if len(distributions_path.split(os.sep)) > 1 else None

    flags_to_check = ["dataset_len", "vocab_family", "crop_to_size"]

    if distributions_dataset_name != dataset_name:
        raise ValueError(f"Dataset name mismatch!")

    for flag in flags_to_check:
        if student_metadata[flag] != distr_metadata[flag]:
            raise ValueError(f"{flag} mismatch!\nStudent metadata: {student_metadata[flag]}\nDistributions metadata: {distr_metadata[flag]}")
    
def finetune(parameters):
    if training.lower() == "full":
        full_finetune(parameters)
    elif training.lower() == "lora":
        lora_finetune(parameters)
    elif training.lower() == "qlora":
        qlora_finetune(parameters)
    else:
        raise Exception("Invalid training type!")

# Main Script
model_path = r"C:\Users\gololo\Desktop\TinyLlama-1.1B-intermediate-step-1431k-3T_safetensors"
dataset_path = r"F:\soup.jsonl"
distributions_path = r"F:\distilled\soup\MythoMax-L2-Kimiko-v2-13b"
save_folder = r"F:\trained"
trained_model_name = "TinyGoblin-1.1B-V3.1"

training = "full" # "full" "lora" "qlora"
optimizer = "adamw" # "adamw8bit" "adamw" "adagrad8bit" "sgd" "paged_adamw8bit" "adabelief"
load_in_half = True
context_length = 2*1024
shuffle_data = True

grad_accumulation_steps = 2
num_epochs = 3
num_warmup_steps = 500
lr = 1e-5
temperature = 1
lr_scheduler = "wsd" # "wsd" "cosine" "linear" "constant" "constant_with_warmup" "polynomial" "inverse_sqrt" "step" "cosine_anneal" "one_cycle"
decay_start = 0.9 # wsd, % of total steps

crop_distr_to_size = 32000
device = "cuda:1"
custom_reduction = False
stop_at_convo = None
add_bos = True
save_every_n_epoch = 1
validation_per_steps = 100

prompt_format = {
    'SYS_START': "#System: ",
    'USER_START': "#User: ",
    'ASSISTANT_START': "#Assistant: ",
    'SYS_END': '\n',
    'USER_END': '\n',
    'ASSISTANT_END': '\n'
}

trained_model_folder = os.path.join(save_folder, trained_model_name)

if not os.path.exists(trained_model_folder):
    os.makedirs(trained_model_folder)

distr_metadata = load_metadata(distributions_path)

print(f"Context Len: {context_length}, Num Epochs: {num_epochs}, Num Warmup Steps: {num_warmup_steps}, LR: {lr} {lr_scheduler}, Optimizer: {optimizer}")

if distr_metadata is not None:
    dataset_tokenized, dataset_content_ranges, empty_convo_ids = tokenize_dataset(
        dataset_path, device, model_path, prompt_format, context_length, 
        distr_metadata['save_sys_range'], distr_metadata['save_user_range'], 
        distr_metadata['save_assistant_range'], add_bos=add_bos)
    
    empty_convo_ids = list(set(distr_metadata.get('empty_convo_ids', []) + empty_convo_ids))
    
    student_metadata = {**{'crop_to_size': crop_distr_to_size}, **generate_metadata(model_path, dataset_tokenized, dataset_content_ranges, empty_convo_ids=empty_convo_ids)}

    dataset_tokenized, dataset_content_ranges = filter_empty_conversations(dataset_tokenized, dataset_content_ranges, empty_convo_ids)

    save_dataset_and_metadata(dataset_tokenized, dataset_content_ranges, student_metadata, trained_model_folder)

    #validation_dataset_path = r"F:\ppl_test_dataset.jsonl"
    #validation_distributions_path = os.path.join(distributions_path, "validate")
    #validation_dataset_tokenized, validation_dataset_content_ranges, validation_empty_convo_ids = tokenize_dataset(
        #validation_dataset_path, device, model_path, prompt_format, context_length,
        #distr_metadata['save_sys_range'], distr_metadata['save_user_range'],
        #distr_metadata['save_assistant_range'], add_bos=add_bos, print_stats=False)
    
    #validation_dataset_tokenized, validation_dataset_content_ranges = filter_empty_conversations(validation_dataset_tokenized, validation_dataset_content_ranges, validation_empty_convo_ids)
    
    check_errors(dataset_path, distributions_path, student_metadata, distr_metadata)

    params = Params()
    params.model_path = model_path
    params.model_name = trained_model_name
    params.save_folder = trained_model_folder
    params.dataset_tokenized = dataset_tokenized
    params.dataset_content_ranges = dataset_content_ranges
    #params.validation_dataset_tokenized = validation_dataset_tokenized
    #params.validation_dataset_content_ranges = validation_dataset_content_ranges
    params.distributions_path = distributions_path
    #params.validation_distributions_path = validation_distributions_path
    params.empty_convo_ids = empty_convo_ids
    #params.validation_empty_convo_ids = validation_empty_convo_ids
    params.validation_per_steps = validation_per_steps
    params.training_type = training
    params.save_interval = save_every_n_epoch
    params.load_in_half = load_in_half
    params.shuffle_data = shuffle_data
    params.optimizer_name = optimizer
    params.context_length = context_length
    params.stop_at_convo = stop_at_convo
    params.full_dataset_len = distr_metadata["dataset_len"]
    params.grad_accum_steps = grad_accumulation_steps
    params.num_epochs = num_epochs
    params.num_warmup_steps = num_warmup_steps
    params.student_metadata = student_metadata
    params.lr = lr
    params.temperature = temperature
    params.decay_start = decay_start
    params.lr_scheduler_name = lr_scheduler
    params.custom_reduction = custom_reduction
    params.device = device
    params.crop_to_size = crop_distr_to_size

    finetune(params)

