import torch
import bitsandbytes as bnb
from transformers import AutoModelForCausalLM, get_scheduler, AutoTokenizer
from tqdm import tqdm
from utils.finetuning_utils import calculate_kl_divergence, teacher_tensors_hander

def full_finetune(parameters):
    dataset_tokenized = parameters["dataset_tokenized"]
    dataset_content_ranges = parameters["dataset_content_ranges"]

    model = AutoModelForCausalLM.from_pretrained(parameters["model_path"], device_map="auto", load_in_8bit=parameters["load_in_8bit"])
    model.train()
    teacher_tensor = teacher_tensors_hander(parameters["distributions_path"], parameters["device"])
    dataset_len = len(dataset_tokenized)
    num_training_steps = parameters["num_epochs"] * len(dataset_tokenized)
    optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=parameters["lr"], betas=(0.9, 0.995))

    lr_scheduler = get_scheduler(name="linear", optimizer=optimizer, num_warmup_steps=parameters["num_warmup_steps"], num_training_steps=num_training_steps)
    recent_losses = []
    pbar = tqdm(total=num_training_steps, desc="Finetuning", unit="convo", smoothing=0.06)

    for step in range(num_training_steps):
        conversation_tokenized, conversation_content_ranges = dataset_tokenized[step % dataset_len], dataset_content_ranges[step % dataset_len]
        conversation_tokenized = conversation_tokenized.to(parameters["device"])[:parameters["context_length"]].unsqueeze(0)
        full_student_logits = model(conversation_tokenized).logits.squeeze(0)
        student_logits = torch.cat([full_student_logits[start:end] for start, end in conversation_content_ranges], dim=0)

        teacher_logits = next(teacher_tensor)

        crop_to_len = min(student_logits.size(0), teacher_logits.size(0))

        loss = calculate_kl_divergence(student_logits[:crop_to_len], teacher_logits[:crop_to_len])
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
        pbar.set_postfix({"Epoch": f"{step / dataset_len:.2f}/{parameters['num_epochs']}", "Avg Dataset Loss": avg_recent_loss})
        pbar.update()
    pbar.close()
    print("Saving model...")
    model.save_pretrained(parameters["save_folder"])
    tokenizer = AutoTokenizer.from_pretrained(parameters["model_path"])
    tokenizer.save_pretrained(parameters["save_folder"])
    print(f"Model successfully saved at {parameters['save_folder']}")