# transformers/main.py
import math
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
from transformers import BertTokenizer
from torch.cuda.amp import GradScaler, autocast
from transformers import get_linear_schedule_with_warmup
from transformer import TransformerModel
from torch.utils.tensorboard import SummaryWriter
from utils.device_setup import device
from utils.early_stopping import EarlyStopping
from utils.evaluation import evaluate_model
from utils.process_data import process_data, create_padding_mask

class TransformerForClassification(TransformerModel):
    def __init__(self, input_dim, embed_dim, num_heads, num_layers, ff_dim, dropout, num_class):
        super(TransformerForClassification, self).__init__(input_dim, input_dim, embed_dim, num_heads, num_layers,
                                                           ff_dim, dropout)
        self.fc = nn.Linear(embed_dim, num_class)

    def forward(self, src, tgt=None, src_mask=None, tgt_mask=None, memory_mask=None):
        src = self.src_embedding(src) * math.sqrt(self.src_embedding.embedding_dim)
        src = self.positional_encoding(src)
        src = self.dropout(src)

        enc_output = self.encoder(src, src_mask)
        enc_output = enc_output.mean(dim=1)
        output = self.fc(enc_output)
        return output

def train_model(model, train_dataloader, val_dataloader, num_epochs, pad_token, save_path='best_model.pth'):
    optimizer = optim.AdamW(model.parameters(), lr=2e-5)
    criterion = nn.BCEWithLogitsLoss()
    scaler = GradScaler()
    writer = SummaryWriter(log_dir='runs/')
    early_stopping = EarlyStopping(patience=10, delta=0)
    total_steps = len(train_dataloader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=0, num_training_steps=total_steps)

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        optimizer.zero_grad()
        for step, batch in enumerate(tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{num_epochs}")):
            texts, masks, labels = batch[0].to(device), batch[1].to(device), batch[2].to(device)
            src_mask = create_padding_mask(texts, pad_token)
            with autocast():
                outputs = model(texts, src_mask=src_mask)
                loss = criterion(outputs, labels)
                loss = loss / 1  # accumulation_steps

            scaler.scale(loss).backward()

            if (step + 1) % 1 == 0:  # accumulation_steps
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
                scheduler.step()

            total_loss += loss.item() * 1  # accumulation_steps

        avg_loss = total_loss / len(train_dataloader)

        val_loss, val_acc = evaluate_model(model, val_dataloader, criterion, model_type='transformer')

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

    embed_dim = 768
    num_heads = 12
    num_layers = 12
    ff_dim = 3072
    dropout = 0.1

    model = TransformerForClassification(input_dim=tokenizer.vocab_size, embed_dim=embed_dim, num_heads=num_heads, num_layers=num_layers, ff_dim=ff_dim, dropout=dropout, num_class=num_class).to(device)

    train_model(model, train_dataloader, val_dataloader, num_epochs, pad_token=tokenizer.pad_token_id)

    test_loss, test_acc = evaluate_model(model, test_dataloader, nn.BCEWithLogitsLoss(), model_type='transformer')
    print(f"Test Loss: {test_loss:.4f}, Test Accuracy: {test_acc:.4f}")