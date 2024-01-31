import math
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, get_scheduler
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from utils.finetuning_utils import calculate_kl_divergence, set_optimizer, scale_temperature
from utils.dataset_utils import H5Reader

def full_finetune(params):
    model = AutoModelForCausalLM.from_pretrained(
        params["model_path"],
        device_map="cuda:1",
        torch_dtype=torch.float16 if params["load_in_half"] else torch.float32
    )
    
    model.train()
    grad_accum_steps = params["grad_accum_steps"]
    teacher = H5Reader(params["distributions_path"], params["device"], empty_convo_ids=params["empty_convo_ids"], timeout=999999)
    optimizer = set_optimizer(
        model.parameters(), 
        lr=params["lr"],
        grad_accum_steps=grad_accum_steps,
        betas=(0.9, 0.995),
        optimizer_name=params["optimizer_name"]
    )

    dataset_len = len(params["dataset_tokenized"])
    num_training_steps = params["num_epochs"] * dataset_len if not params["stop_at_convo"] else params["stop_at_convo"]
    num_grad_updates = math.ceil(num_training_steps / grad_accum_steps)
    lr_scheduler = get_scheduler(
        name=params["lr_scheduler_name"].lower(),
        optimizer=optimizer, 
        num_warmup_steps=params["num_warmup_steps"], 
        num_training_steps=num_grad_updates
    )

    print(f"Num Gradient Updates: {num_grad_updates}")
    recent_losses = []
    updated = False
    logger = SummaryWriter(log_dir=params["save_folder"], comment=f"_{params['training_type']}_finetune__lr_{params['lr']}")
    pbar = tqdm(total=num_training_steps, desc=f"{params['training_type']} Finetuning", unit="convo", smoothing=0.06)
    
    for step in range(num_training_steps):
        updated = False
        temp = scale_temperature(step, num_training_steps, params["temperature"])
        idx = step % dataset_len
        conversation_tokenized, conversation_content_ranges = params["dataset_tokenized"][idx], params["dataset_content_ranges"][idx]
        teacher_probs = teacher.read_next()
        min_len = min(teacher_probs.size(0), len([conversation_tokenized[:params["context_length"]][start:end] for start, end in conversation_content_ranges]))
        conversation_tokenized = conversation_tokenized.to(params["device"])[:params["context_length"]].unsqueeze(0)
        teacher_probs = teacher_probs[:min_len]
        
        if params["per_token_training"]:
            total_loss = 0
            pbar2 = tqdm(total=min_len, desc=f"Per-Token Training", unit="token", smoothing=0.05, leave=False)
            for current_token_id in range(min_len):
                full_student_logits = model(conversation_tokenized).logits.squeeze(0)
                student_logits = torch.cat([full_student_logits[start:end] for start, end in conversation_content_ranges], dim=0)[:, :params["crop_to_size"]]
                loss = calculate_kl_divergence(student_logits[current_token_id], teacher_probs[current_token_id], temp=temp, per_token=True)
                loss.backward()

                recent_losses.append(loss.item())
                optimizer.step()
                optimizer.zero_grad()
                updated = True
                total_loss += loss.item()
                pbar2.set_postfix({"Loss": total_loss / (current_token_id + 1)})
                pbar2.update()
            lr_scheduler.step()
            pbar2.close()
        else:
            full_student_logits = model(conversation_tokenized).logits.squeeze(0)
            student_logits = torch.cat([full_student_logits[start:end] for start, end in conversation_content_ranges], dim=0)[:min_len][:, :params["crop_to_size"]]
            loss = calculate_kl_divergence(student_logits, teacher_probs, temp=temp)
            loss.backward()

            recent_losses.append(loss.item())
            if (step + 1) % grad_accum_steps == 0:
                optimizer.step()
                lr_scheduler.step()
                optimizer.zero_grad()
                updated = True
        
        avg_loss = sum(recent_losses) / len(recent_losses)
        if len(recent_losses) > 100:
            recent_losses = recent_losses[-99:]

        logger.add_scalar('Loss/train', avg_loss, step)
        logger.add_scalar('Learning Rate', lr_scheduler.get_last_lr()[0], step)
        logger.add_scalar('Temperature', temp, step)

        pbar.set_postfix({"Epoch": f"{(step+1) / dataset_len:.2f}/{params['num_epochs']}", "Loss": avg_loss, "LR": lr_scheduler.get_last_lr()[0]})
        pbar.update()
    
    if not updated:
        optimizer.step()
        optimizer.zero_grad()

    logger.close()
    pbar.close()
    teacher.close()
    model.save_pretrained(params["save_folder"])
    tokenizer = AutoTokenizer.from_pretrained(params["model_path"])
    tokenizer.save_pretrained(params["save_folder"])
    print(f"Model successfully saved at {params['save_folder']}")