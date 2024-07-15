# bert/main.py
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import BertTokenizer, BertForSequenceClassification
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from torch.utils.tensorboard import SummaryWriter
from utils.device_setup import device
from utils.early_stopping import EarlyStopping
from utils.evaluation import evaluate_model
from utils.process_data import process_data


def train_model(model, train_dataloader, val_dataloader, num_epochs, save_path='best_model.pth'):
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    writer = SummaryWriter(log_dir='runs/')
    early_stopping = EarlyStopping(patience=10, delta=0)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    train_losses = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            inputs = batch[0].to(device)
            attention_mask = batch[1].to(device)
            labels = batch[2].to(device)

            with autocast():
                outputs = model(input_ids=inputs, attention_mask=attention_mask)
                loss = criterion(outputs.logits, labels)
                loss = loss / 1  # accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % 1 == 0:  # accumulation_steps
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * 1  # accumulation_steps

        avg_loss = total_loss / len(train_dataloader)
        train_losses.append(avg_loss)

        val_loss, val_acc = evaluate_model(model, val_dataloader, criterion)
        val_losses.append(val_loss)
        val_accuracies.append(val_acc)

        print(f"Epoch: {epoch + 1} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")

        writer.add_scalar('Loss/train', avg_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        writer.add_scalar('Accuracy/val', val_acc, epoch)

        early_stopping(val_loss)

        if early_stopping.early_stop:
            print(f"Early stopping at epoch {epoch + 1}")
            break

        torch.save(model.state_dict(), save_path)

    writer.close()


if __name__ == "__main__":
    batch_size = 32
    max_len = 250
    num_epochs = 100

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataloader, val_dataloader, test_dataloader, num_class = process_data(
        tokenizer=tokenizer,
        batch_size=batch_size,
        max_len=max_len,
    )

    model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=num_class)
    model.to(device)

    train_model(model, train_dataloader, val_dataloader, num_epochs=num_epochs)

    test_loss, test_acc = evaluate_model(model, test_dataloader, nn.BCEWithLogitsLoss())
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")
