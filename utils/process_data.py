import os
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
import nltk
import matplotlib.pyplot as plt
from collections import Counter
from nltk.corpus import reuters
from transformers import BertTokenizer

# Tải tập dữ liệu Reuters
nltk.download('reuters')

def read_data(p=100):
    fileids = reuters.fileids()
    fileids = fileids[:int(len(fileids) * p / 100)]
    texts = [reuters.raw(doc_id) for doc_id in fileids]
    labels = [reuters.categories(doc_id) for doc_id in fileids]
    return texts, labels

def tokenize_texts(texts, tokenizer, max_len):
    tokenized_data = tokenizer(
        texts,
        max_length=max_len,
        padding='max_length',
        truncation=True,
        return_tensors='pt'
    )
    return tokenized_data

def print_data_stats(texts, labels, name):
    num_samples = len(texts)
    num_labels = [len(label) for label in labels]

    label_counter = Counter(num_labels)
    max_len = max(len(text.split()) for text in texts)
    min_len = min(len(text.split()) for text in texts)
    avg_len = sum(len(text.split()) for text in texts) / len(texts)

    print(f"--- {name} Statistics ---")
    print(f"Number of samples: {num_samples}")
    print(f"Max number of labels: {max(num_labels)}")
    print(f"Min number of labels: {min(num_labels)}")
    print(f"Average number of labels: {sum(num_labels) / num_samples:.2f}")
    print(f"Max length of texts: {max_len}")
    print(f"Min length of texts: {min_len}")
    print(f"Average length of texts: {avg_len:.2f}")
    print("Label distribution:")
    for num, count in label_counter.items():
        print(f"Number of texts with {num} labels: {count} ({count / num_samples:.2%})")
    print()

    # Biểu đồ phân phối số lượng nhãn trên mỗi văn bản
    plt.figure(figsize=(10, 5))
    plt.bar(label_counter.keys(), label_counter.values())
    plt.xlabel('Number of Labels')
    plt.ylabel('Number of Texts')
    plt.title(f'Label Distribution in {name}')
    plt.show()

    # Biểu đồ phân phối độ dài các văn bản
    text_lengths = [len(text.split()) for text in texts]
    plt.figure(figsize=(10, 5))
    plt.hist(text_lengths, bins=50, edgecolor='k')
    plt.xlabel('Text Length')
    plt.ylabel('Number of Texts')
    plt.title(f'Text Length Distribution in {name}')
    plt.show()

def process_data(tokenizer, batch_size=32, p=100, max_len=128):
    texts, labels = read_data(p)

    # In thống kê chi tiết về dữ liệu
    print_data_stats(texts, labels, "Full Dataset")

    # Tokenize các câu sử dụng tokenizer
    tokenized_data = tokenize_texts(texts, tokenizer, max_len)

    input_ids = tokenized_data['input_ids']
    attention_masks = tokenized_data['attention_mask']

    # Sử dụng MultiLabelBinarizer để chuyển đổi nhãn thành nhãn nhị phân
    mlb = MultiLabelBinarizer()
    label_tensor = torch.tensor(mlb.fit_transform(labels), dtype=torch.float)

    # Chia dữ liệu thành tập huấn luyện, kiểm tra và xác thực
    train_inputs, temp_inputs, train_labels, temp_labels = train_test_split(input_ids, label_tensor, test_size=0.3, random_state=42)
    train_masks, temp_masks = train_test_split(attention_masks, test_size=0.3, random_state=42)

    val_inputs, test_inputs, val_labels, test_labels = train_test_split(temp_inputs, temp_labels, test_size=0.5, random_state=42)
    val_masks, test_masks = train_test_split(temp_masks, test_size=0.5, random_state=42)

    # Tạo DataLoader
    train_dataset = TensorDataset(train_inputs, train_masks, train_labels)
    val_dataset = TensorDataset(val_inputs, val_masks, val_labels)
    test_dataset = TensorDataset(test_inputs, test_masks, test_labels)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    num_class = label_tensor.shape[1]

    return train_dataloader, val_dataloader, test_dataloader, num_class

if __name__ == "__main__":
    data_name = 'reuters'
    batch_size = 32
    p = 100
    max_len = 128

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    train_dataloader, val_dataloader, test_dataloader, num_class = process_data(
        tokenizer=tokenizer,
        batch_size=batch_size,
        p=p,
        max_len=max_len
    )

    print(num_class)



def create_padding_mask(seq, pad_token=0):
    mask = (seq != pad_token).unsqueeze(1).unsqueeze(2)
    return mask
