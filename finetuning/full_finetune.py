import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from tqdm import tqdm
from utils.finetuning_utils import calculate_kl_divergence, teacher_tensors_hander, set_optimizer

def full_finetune(params):
    model = AutoModelForCausalLM.from_pretrained(
        params["model_path"],
        device_map="auto",
        load_in_8bit=params["load_in_8bit"]
    )
    
    model.train()
    grad_accum_steps = params["grad_accum_steps"]
    teacher_tensor = teacher_tensors_hander(params["distributions_path"], params["device"])
    optimizer = set_optimizer(
        model.parameters(), 
        lr=params["lr"],
        grad_accum_steps=grad_accum_steps,
        betas=(0.9, 0.995),
        optimizer_name=params["optimizer_name"]
    )

    dataset_len = len(params["dataset_tokenized"])
    num_training_steps = params["num_epochs"] * dataset_len
    lr_scheduler = get_scheduler(
        name=params["lr_scheduler_name"].lower(),
        optimizer=optimizer, 
        num_warmup_steps=params["num_warmup_steps"], 
        num_training_steps=len(params["dataset_tokenized"])
    )

    print("Num Gradient Updates:", math.ceil(dataset_len / grad_accum_steps))
    recent_losses = []
    updated = False
    pbar = tqdm(total=num_training_steps, desc=f"{params['training_type']} Finetuning", unit="convo", smoothing=0.06)
    for step in range(num_training_steps):
        updated = False
        idx = step % dataset_len
        conversation_tokenized, conversation_content_ranges = params["dataset_tokenized"][idx], params["dataset_content_ranges"][idx]
        conversation_tokenized = conversation_tokenized.to(params["device"])[:params["context_length"]].unsqueeze(0)

        full_student_logits = model(conversation_tokenized).logits.squeeze(0)
        student_logits = torch.cat([full_student_logits[start:end] for start, end in conversation_content_ranges], dim=0)

        teacher_logits = next(teacher_tensor)
        min_len = min(student_logits.size(0), teacher_logits.size(0))
        loss = calculate_kl_divergence(student_logits[:min_len], teacher_logits[:min_len])

        loss.backward()

        if (step + 1) % grad_accum_steps == 0:
            updated = True
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        recent_losses.append(loss.item())
        if len(recent_losses) > dataset_len:
            recent_losses.pop(0)
        avg_loss = sum(recent_losses) / len(recent_losses)

        pbar.set_postfix({"Epoch": f"{(step+1) / dataset_len:.2f}/{params['num_epochs']}", "Avg Loss": avg_loss, "LR": lr_scheduler.get_last_lr()[0]})
        pbar.update()
    
    if not updated:
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad()

    pbar.close()

    model.save_pretrained(params["save_folder"])
    tokenizer = AutoTokenizer.from_pretrained(params["model_path"])
    tokenizer.save_pretrained(params["save_folder"])
    print(f"Model successfully saved at {params['save_folder']}")