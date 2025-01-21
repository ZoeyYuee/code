import csv
import os

import gensim
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

def read_data(filename, num=None):
    texts = []
    labels = []

    with open(filename, encoding="utf-8") as csv_file:
        csv_reader = csv.DictReader(csv_file)
        next(csv_reader)

        for row in csv_reader:
            if len(row) >= 8:
                text = row["comment"]
                label = int(row["state"])

                texts.append(text)
                labels.append(label)

    return texts[:num], labels[:num]


def built_curpus(train_texts, embedding_num):
    word_2_index = {"<PAD>": 0, "<UNK>": 1}
    for text in train_texts:
        for word in text.split():
            word_2_index[word] = word_2_index.get(word, len(word_2_index))
    return word_2_index, nn.Embedding(len(word_2_index), embedding_num)


class TextDataset(Dataset):
    def __init__(self, texts, labels, word_2_index, index_2_embedding, max_len):
        self.texts = texts
        self.labels = labels
        self.word_2_index = word_2_index
        self.index_2_embedding = index_2_embedding
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.texts[index]
        label = int(self.labels[index])
        text = text[:self.max_len]
        text_index = [word_2_index.get(i, 1) for i in text]
        text_index = text_index + [0] * (self.max_len - len(text_index))
        text_onehot = self.index_2_embedding(torch.tensor(text_index))
        return text_onehot, label

    def __len__(self):
        return len(self.labels)


class RNNModel(nn.Module):
    def __init__(self, embedding_num, hidden_num, class_num, max_len):
        super().__init__()
        self.rnn = nn.RNN(embedding_num, hidden_num, num_layers=1, bias=True, batch_first=True, bidirectional=False)
        self.classifier = nn.Linear(max_len * hidden_num, class_num)
        self.cross_loss = nn.CrossEntropyLoss()

    def forward(self, text_embedding, labels=None):
        out, h_n = self.rnn(text_embedding, None)
        out = out.reshape(out.shape[0], -1)
        p = self.classifier(out)
        self.pre = torch.argmax(p, dim=-1).detach().cpu().numpy().tolist()
        if labels is not None:
            loss = self.cross_loss(p, labels)
            return loss


TRAIN_PATH = "1_trian_data.csv"
TEST_PATH = "1_test_data.csv"


if __name__ == "__main__":
    train_texts, train_labels = read_data(TRAIN_PATH)
    dev_texts, dev_labels = read_data(TEST_PATH)

    assert len(train_texts) == len(train_labels)
    assert len(dev_texts) == len(dev_labels)

    epoch = 5
    batch_size = 60
    max_len = 25
    hidden_num = 300
    embedding_num = 200
    lr = 0.0006

    class_num = len(set(train_labels))
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    word_2_index, index_2_embedding = built_curpus(train_texts, embedding_num)

    train_dataset = TextDataset(train_texts, train_labels, word_2_index, index_2_embedding, max_len)
    train_dataloader = DataLoader(train_dataset, batch_size, shuffle=False)

    dev_dataset = TextDataset(dev_texts, dev_labels, word_2_index, index_2_embedding, max_len)
    dev_dataloader = DataLoader(dev_dataset, batch_size, shuffle=False)

    model = RNNModel(embedding_num, hidden_num, class_num, max_len)
    model = model.to(device)
    optim = torch.optim.AdamW(model.parameters(), lr=lr)

    for e in range(epoch):
        for texts, labels in tqdm(train_dataloader):
            texts = texts.to(device)
            labels = labels.to(device)

            loss = model(texts, labels)
            loss.backward()

            optim.step()
            optim.zero_grad()

        right_num = 0
        total_samples = 0
        true_labels = []
        predicted_labels = []
        for texts, labels in dev_dataloader:
            texts = texts.to(device)
            model(texts)
            true_labels.extend(labels.tolist())
            predicted_labels.extend(model.pre)
            right_num += int(sum([i == j for i, j in zip(model.pre, labels)]))
        print(f"dev acc : {right_num / len(dev_labels) * 100 : .2f}%")

        precision = precision_score(true_labels, predicted_labels, average='weighted')
        recall = recall_score(true_labels, predicted_labels, average='weighted')
        f1 = f1_score(true_labels, predicted_labels, average='weighted')

        print(f"Epoch {e + 1} - Dev Accuracy: {right_num / len(dev_labels) * 100:.2f}%")
        print(f"Precision: {precision:.4f}, Recall: {recall:.4f}, F1-Score: {f1:.4f}")
