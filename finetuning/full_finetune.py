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
        device_map="auto",
        torch_dtype=torch.float16 if params["load_in_half"] else torch.float32
    )
    
    model.train()
    grad_accum_steps = params["grad_accum_steps"]
    teacher = H5Reader(params["distributions_path"], params["device"], empty_convo_ids=params["empty_convo_ids"])
    optimizer = set_optimizer(
        model.parameters(), 
        lr=params["lr"],
        grad_accum_steps=grad_accum_steps,
        betas=(0.9, 0.995),
        optimizer_name=params["optimizer_name"]
    )

    dataset_len = len(params["dataset_tokenized"])
    num_training_steps = params["num_epochs"] * dataset_len
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
    logger = SummaryWriter(log_dir=params["save_folder"])
    pbar = tqdm(total=num_training_steps, desc=f"{params['training_type']} Finetuning", unit="convo", smoothing=0.06)
    
    for step in range(num_training_steps):
        updated = False
        temp = scale_temperature(step, num_training_steps, params["temperature"])
        idx = step % dataset_len
        conversation_tokenized, conversation_content_ranges = params["dataset_tokenized"][idx], params["dataset_content_ranges"][idx]
        conversation_tokenized = conversation_tokenized.to(params["device"])[:params["context_length"]].unsqueeze(0)

        full_student_logits = model(conversation_tokenized).logits.squeeze(0)
        student_logits = torch.cat([full_student_logits[start:end] for start, end in conversation_content_ranges], dim=0)[:, :params["clip_to_size"]]

        teacher_probs = teacher.read_next()
        min_len = min(student_logits.size(0), teacher_probs.size(0))
        loss = calculate_kl_divergence(student_logits[:min_len], teacher_probs[:min_len], temp=temp)
        
        loss.backward()

        logger.add_scalar('Loss/train', loss.item(), step)
        logger.add_scalar('Learning Rate', lr_scheduler.get_last_lr()[0], step)

        if torch.isnan(loss).any() or torch.isinf(loss).any():
            print(f"Warning: NaN or Inf found in loss at step {step}")
            lr_scheduler.step()
            optimizer.zero_grad()
            continue

        if (step + 1) % grad_accum_steps == 0:
            updated = True
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()

        recent_losses.append(loss.item())
        if len(recent_losses) > 100:
            recent_losses.pop(0)
        avg_loss = sum(recent_losses) / len(recent_losses)

        pbar.set_postfix({"Epoch": f"{(step+1) / dataset_len:.2f}/{params['num_epochs']}", "Avg 100 Loss": avg_loss, "LR": lr_scheduler.get_last_lr()[0]})
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