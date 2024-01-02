import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from tqdm import tqdm
from utils.finetuning_utils import calculate_kl_divergence, teacher_tensors_hander, set_optimizer

def full_finetune(parameters):
    model = AutoModelForCausalLM.from_pretrained(
        parameters["model_path"], 
        device_map="auto", 
        load_in_8bit=parameters["load_in_8bit"]
    )
    
    model.train()
    teacher_tensor = teacher_tensors_hander(parameters["distributions_path"], parameters["device"])
    optimizer = set_optimizer(
        model.parameters(), 
        lr=parameters["lr"], 
        betas=(0.9, 0.995), 
        optimizer_name=parameters["optimizer_name"]
    )

    num_training_steps = parameters["num_epochs"] * len(parameters["dataset_tokenized"])
    lr_scheduler = get_scheduler(
        name=parameters["lr_scheduler_name"].lower(), 
        optimizer=optimizer, 
        num_warmup_steps=parameters["num_warmup_steps"], 
        num_training_steps=num_training_steps
    )

    recent_losses = []
    pbar = tqdm(total=num_training_steps, desc="Finetuning", unit="convo", smoothing=0.06)
    for step in range(num_training_steps):
        idx = step % len(parameters["dataset_tokenized"])
        conversation_tokenized, conversation_content_ranges = parameters["dataset_tokenized"][idx], parameters["dataset_content_ranges"][idx]
        conversation_tokenized = conversation_tokenized.to(parameters["device"])[:parameters["context_length"]].unsqueeze(0)
        
        full_student_logits = model(conversation_tokenized).logits.squeeze(0)
        student_logits = torch.cat([full_student_logits[start:end] for start, end in conversation_content_ranges], dim=0)

        teacher_logits = next(teacher_tensor)
        min_len = min(student_logits.size(0), teacher_logits.size(0))
        loss = calculate_kl_divergence(student_logits[:min_len], teacher_logits[:min_len])

        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

        recent_losses.append(loss.item())
        if len(recent_losses) > len(parameters["dataset_tokenized"]):
            recent_losses.pop(0)
        avg_loss = sum(recent_losses) / len(recent_losses)
        pbar.set_postfix({"Epoch": f"{(step+1) / len(parameters['dataset_tokenized']):.2f}/{parameters['num_epochs']}", "Avg Loss": avg_loss})
        pbar.update()
    pbar.close()

    model.save_pretrained(parameters["save_folder"])
    tokenizer = AutoTokenizer.from_pretrained(parameters["model_path"])
    tokenizer.save_pretrained(parameters["save_folder"])
    print(f"Model successfully saved at {parameters['save_folder']}")