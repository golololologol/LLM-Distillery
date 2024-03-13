import math
import torch
import os
from datetime import datetime
from transformers import AutoModelForCausalLM, AutoTokenizer
from torch.utils.tensorboard.writer import SummaryWriter
from tqdm import tqdm
from utils.finetuning_utils import calculate_divergence, set_optimizer, set_lr_scheduler, Params, launch_tensorboard, scale_temperature
from utils.dataset_utils import H5Reader

def save_model(model, model_path, model_name, save_folder, epoch):
    save_folder = os.path.join(save_folder, f"{model_name}_epoch_{epoch}")
    model.save_pretrained(save_folder)
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    tokenizer.save_pretrained(save_folder)

def full_finetune(params: Params):
    device = params.device
    model = AutoModelForCausalLM.from_pretrained(
        params.model_path,
        device_map=device,
        torch_dtype=torch.float16 if params.load_in_half else torch.float32
    )
    
    model.train()
    grad_accum_steps = params.grad_accum_steps
    optimizer = set_optimizer(
        model.parameters(), 
        lr=params.lr,
        grad_accum_steps=grad_accum_steps,
        betas=(0.5, 0.9),
        optimizer_name=params.optimizer_name,
        weight_decay=2e-5
    )

    dataset_len = len(params.dataset_tokenized)
    num_training_steps = params.num_epochs * dataset_len if not params.stop_at_convo else params.stop_at_convo
    num_grad_updates = math.ceil(num_training_steps / grad_accum_steps)
    lr_scheduler = set_lr_scheduler(
        optimizer, 
        lr_scheduler_name=params.lr_scheduler_name, 
        num_warmup_steps=params.num_warmup_steps, 
        num_training_steps=num_training_steps, 
        num_epoch_steps=dataset_len,
        decay_start=params.decay_start
    )

    #val_teacher = H5Reader(params.validation_distributions_path, device, empty_convo_ids=params.validation_empty_convo_ids, shuffle=False)
    #val_tensors = [val_teacher.read_next() for i in range(val_teacher.dataset_len - len(params.validation_empty_convo_ids))]
    #val_teacher.close()
    
    def get_content_indices(content_ranges, context_len):
        content_indices = []
        for start, end in content_ranges:
            if start <= context_len:
                content_indices.append(torch.arange(start, min(end, context_len), device=device))
        return torch.cat(content_indices)
    
    #def validate(model, dataset_tokenized, dataset_content_ranges, device, teacher_loader: H5Reader, custom=False, val_tensors=val_tensors):
        val_loss = 0
        teacher_loader.toggle_await()
        with torch.no_grad():
            for convo_tokenized, convo_content_ranges, val_tensor in zip(dataset_tokenized, dataset_content_ranges, val_tensors):
                convo_tokenized = convo_tokenized.to(device)[:params.context_length].unsqueeze(0)
                content_indices = get_content_indices(convo_content_ranges, params.context_length)
                full_student_logits = model(convo_tokenized).logits.squeeze(0).float()
                student_logits = torch.index_select(full_student_logits, 0, content_indices)[:, :params.crop_to_size]
                min_len = min(student_logits.size(0), val_tensors[0].size(0))
                val_loss += calculate_divergence(student_logits[:min_len], val_tensor[:min_len], custom=params.custom_reduction, temp=1).item()

        teacher_loader.toggle_await()
        return val_loss / len(val_tensors)
        
    teacher = H5Reader(params.distributions_path, device, empty_convo_ids=params.empty_convo_ids, shuffle=params.shuffle_data, timeout=90)
    print(f"Num Gradient Updates: {num_grad_updates}")
    recent_losses = []
    order_list = []
    filtered_order_list = []
    updated = False

    run_id = datetime.now().strftime("%Y%m%d-%H%M%S") + f"_{params.training_type}_lr_{params.lr}"
    log_dir = os.path.join(params.save_folder, "tensorboard_logs", run_id)
    os.makedirs(log_dir, exist_ok=True)
    logger = SummaryWriter(log_dir=log_dir)
    tensorboard = launch_tensorboard(params.save_folder)

    pbar = tqdm(range(num_training_steps), desc=f"{params.training_type} Finetuning", unit="convo", smoothing=0.06)
    dataset_tokenized = params.dataset_tokenized
    dataset_content_ranges = params.dataset_content_ranges

    for step in pbar:
        updated = False
        pbar.update
        if params.shuffle_data:
            if not filtered_order_list:
                order_list = torch.randperm(params.full_dataset_len).tolist()
                teacher.set_shuffle_order(order_list)
                filtered_order_list = [i - sum(j < i for j in params.empty_convo_ids) for i in order_list if i not in params.empty_convo_ids]
            idx = filtered_order_list.pop(0)

        else:
            idx = step % dataset_len

        conversation_tokenized, conversation_content_ranges = dataset_tokenized[idx], dataset_content_ranges[idx]
        teacher_log_probs = teacher.read_next()
        conversation_tokenized = conversation_tokenized.to(device)[:params.context_length].unsqueeze(0)

        content_indices = get_content_indices(conversation_content_ranges, params.context_length)
        full_student_logits = model(conversation_tokenized).logits.squeeze(0).float()
        
        student_logits = torch.index_select(full_student_logits, 0, content_indices)[:, :params.crop_to_size]
        min_len = min(student_logits.size(0), teacher_log_probs.size(0))

        temp = scale_temperature(step, num_training_steps, params.temperature)
        loss = calculate_divergence(student_logits[:min_len], teacher_log_probs[:min_len], custom=params.custom_reduction, temp=temp)
        loss.backward()

        recent_losses.append(loss.item())
        if (step + 1) % grad_accum_steps == 0:
            optimizer.step() # type: ignore
            optimizer.zero_grad() # type: ignore
            updated = True

        if len(recent_losses) > 100:
            recent_losses.pop(0)
        avg_loss = sum(recent_losses) / len(recent_losses)

        lr_scheduler.step() # type: ignore
        
        logger.add_scalar('Loss/train', loss.item(), step)
        logger.add_scalar('Learning Rate', lr_scheduler.get_last_lr()[0], step) # type: ignore
        #if (step + 1) % params.validation_per_steps == 0 or step == 0:
            #val_loss = validate(model, params.validation_dataset_tokenized, params.validation_dataset_content_ranges, device, teacher, val_tensors=val_tensors)
            #logger.add_scalar('Loss/val', val_loss, step)

        pbar.set_postfix({"Epoch": f"{(step+1) / dataset_len:.2f}/{params.num_epochs}", "Loss": avg_loss, "LR": lr_scheduler.get_last_lr()[0]}) # type: ignore
        
        if ((step + 1) % (params.save_interval * dataset_len) == 0) and (step + 1) != num_training_steps:
            save_model(model, params.model_path, params.model_name, params.save_folder, (step + 1) // dataset_len)

    if not updated:
        optimizer.step() # type: ignore

    save_model(model, params.model_path, params.model_name, params.save_folder, "final")
    logger.close()
    pbar.close()
    teacher.close()
    tensorboard.terminate()

    print(f"Model successfully saved at {params.save_folder}")