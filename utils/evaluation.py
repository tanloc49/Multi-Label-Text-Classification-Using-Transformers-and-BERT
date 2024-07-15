import torch
from torch import nn
from transformers.modeling_outputs import SequenceClassifierOutput

from utils.device_setup import device
from utils.process_data import create_padding_mask


def evaluate_model(model, dataloader, criterion, model_type=None):
    model.eval()
    total_loss = 0
    total_correct = 0

    with torch.no_grad():
        for batch in dataloader:
            if model_type == 'transformer':
                inputs = batch[0].to(device)
                attention_mask = None  # No attention mask needed for transformer
            else:
                inputs = batch[0].to(device)
                attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            if model_type == 'transformer':
                outputs = model(inputs, src_mask=create_padding_mask(inputs, pad_token=0))
            else:
                outputs = model(input_ids=inputs, attention_mask=attention_mask)

            if isinstance(outputs, SequenceClassifierOutput):
                logits = outputs.logits
            elif isinstance(outputs, tuple):
                logits = outputs[0]
            else:
                logits = outputs

            loss = criterion(logits, labels)
            total_loss += loss.item()

            preds = (logits > 0.5).float()
            total_correct += (preds == labels).all(axis=1).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = total_correct / len(dataloader.dataset)
    return avg_loss, accuracy


