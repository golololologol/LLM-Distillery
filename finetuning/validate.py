import torch.nn as nn
import torch

def validate(model, validation_dataset_tokenized, validation_dataset_ranges, device):
    model.eval()
    total_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for conversation_tokenized in validation_dataset_tokenized:
            inputs = conversation_tokenized.to(device)
            outputs = model(inputs, labels=inputs)
            logits = outputs.logits

            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = inputs[..., 1:].contiguous()

            loss = criterion(shift_logits.view(-1, shift_logits.size(-1)), shift_labels.view(-1))
            total_loss += loss.item()

    average_loss = total_loss / len(validation_dataset_tokenized)
    return average_loss