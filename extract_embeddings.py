# extract_embeddings.py
from transformers import BertTokenizer, BertModel
import torch
from datasets import load_dataset, concatenate_datasets
import numpy as np
from tqdm import tqdm
import os

def get_embedding_strategy(hidden_states, strategy):
    if strategy == 'first':
        return hidden_states[1]
    elif strategy == 'last':
        return hidden_states[-1]
    elif strategy == 'sum_all':
        return torch.stack(hidden_states, dim=0).sum(0)
    elif strategy == 'second_last':
        return hidden_states[-2]
    elif strategy == 'sum_last4':
        return torch.stack(hidden_states[-4:], dim=0).sum(0)
    elif strategy == 'concat_last4':
        return torch.cat(hidden_states[-4:], dim=-1)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")

def extract_embeddings(dataset_name, split='train', strategy='concat_last4', max_len=128, limit=500):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased', output_hidden_states=True)
    model.eval()

    dataset = load_dataset(dataset_name, split=split)# Ambil data hanya jika mengandung minimal dua kelas
    label_0 = dataset.filter(lambda x: x['label'] == 0).select(range(250))
    label_1 = dataset.filter(lambda x: x['label'] == 1).select(range(250))
    dataset = concatenate_datasets([label_0, label_1])
    texts = []
    labels = []
    seen_labels = set()

    for example in dataset:
        if 'text' not in example:
            # Untuk SNLI, gunakan premis dan hipotesis
            text = example['premise'] + " [SEP] " + example['hypothesis']
        else:
            text = example['text']

        label = example['label']

        # Abaikan label -1 (SNLI sering memiliki label -1 sebagai "tidak valid")
        if label == -1:
            continue

        texts.append(text)
        labels.append(label)
        seen_labels.add(label)

        if len(texts) >= limit and len(seen_labels) > 1:
            break

    all_embeddings = []
    all_labels = []

    for i, (text, label) in enumerate(tqdm(zip(texts, labels), total=len(texts))):
        inputs = tokenizer(text, return_tensors='pt', max_length=max_len, truncation=True, padding='max_length')
        with torch.no_grad():
            outputs = model(**inputs)
            hidden_states = outputs.hidden_states  # tuple of 13 (one for each layer + embedding)
            emb = get_embedding_strategy(hidden_states, strategy)
            cls_embedding = emb[:, 0, :]  # Use [CLS] token embedding
            all_embeddings.append(cls_embedding.squeeze(0).numpy())
            all_labels.append(label)

    os.makedirs("embeddings", exist_ok=True)
    np.save(f"embeddings/{dataset_name}_{strategy}_embeddings.npy", np.array(all_embeddings))
    np.save(f"embeddings/{dataset_name}_{strategy}_labels.npy", np.array(all_labels))
