import torch
import gensim
import csv
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np


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

word2vec_model = gensim.models.Word2Vec.load("your_word2vec_model.model")

def built_curpus(train_texts):
    word_2_index = {"<PAD>": 0, "<UNK>": 1}
    embedding_matrix = [np.zeros(word2vec_model.vector_size), np.random.rand(word2vec_model.vector_size)]
    for text in train_texts:
        for word in text.split():
            if word in word2vec_model.wv:
                if word not in word_2_index:
                    word_2_index[word] = len(word_2_index)
                    embedding_matrix.append(word2vec_model.wv.get_vector(word))
    embedding_matrix = np.array(embedding_matrix)
    embedding_matrix = torch.tensor(embedding_matrix, dtype=torch.float32)
    embedding_matrix = nn.Embedding.from_pretrained(embedding_matrix, padding_idx=0, freeze=False)

    return word_2_index, embedding_matrix

class TextDataset(Dataset):
    def __init__(self, all_text, all_label, word_2_index, max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.word_2_index = word_2_index
        self.max_len = max_len

    def __getitem__(self, index):
        text = self.all_text[index][:self.max_len]
        label = int(self.all_label[index])

        text_idx = [self.word_2_index.get(word, 1) for word in text]
        text_idx = text_idx + [0] * (self.max_len - len(text_idx))
        text_idx = text_idx[:self.max_len]
        text_idx = torch.tensor(text_idx).unsqueeze(dim=0)

        return text_idx, label

    def __len__(self):
        return len(self.all_text)

class CNNModel(nn.Module):
    def __init__(self, max_len, class_num, hidden_num):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=hidden_num, kernel_size=(1, 32)).to(DEVICE, dtype=torch.float32)
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=hidden_num, kernel_size=(1, 64)).to(DEVICE, dtype=torch.float32)
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=hidden_num, kernel_size=(1, 128)).to(DEVICE, dtype=torch.float32)

        self.dropout = nn.Dropout(DROP_PROB)
        self.linear = nn.Linear(hidden_num * 3, class_num)

    def forward(self, batch_idx, batch_label=None):
        batch_idx = batch_idx.unsqueeze(1)
        out1 = F.relu(self.conv1(batch_idx))
        out2 = F.relu(self.conv2(batch_idx))
        out3 = F.relu(self.conv3(batch_idx))

        out1 = F.max_pool2d(out1, (out1.shape[2], out1.shape[3])).squeeze()
        out2 = F.max_pool2d(out2, (out2.shape[2], out2.shape[3])).squeeze()
        out3 = F.max_pool2d(out3, (out3.shape[2], out3.shape[3])).squeeze()

        feature = torch.cat([out1, out2, out3], dim=1)

        feature = self.dropout(feature)
        pred = self.linear(feature)

        return pred


EPOCH = 5
TRAIN_PATH = "1_trian_data.csv"
TEST_PATH = "1_test_data.csv"
BERT_PAD_ID = 0
MAX_LEN = 200
BATH_SIZE = 128
LR = 0.0001
CLASS_NUM = 2
EMBEDDING = 768


OUTPUT_SIZE = 2
N_LAYERS = 2
HIDDEN_DIM = 128
DROP_PROB = 0.2


if __name__ == "__main__":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"


    train_text, train_label = read_data(TRAIN_PATH)
    test_text, test_label = read_data(TEST_PATH)

    word_2_index, words_embedding = built_curpus(train_text)

    train_dataset = TextDataset(train_text, train_label, word_2_index, MAX_LEN)
    train_loader = DataLoader(train_dataset, BATH_SIZE, shuffle=True)

    test_dataset = TextDataset(test_text, test_label, word_2_index, MAX_LEN)
    test_loader = DataLoader(test_dataset, BATH_SIZE, shuffle=True)

    model = CNNModel(MAX_LEN, CLASS_NUM, HIDDEN_DIM).to(DEVICE, dtype=torch.float32)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    for e in range(EPOCH):
        for batch_idx, batch_label in train_loader:
            batch_idx = batch_idx.to(DEVICE, dtype=torch.float32)
            batch_label = batch_label.to(DEVICE)
            train_pred = model.forward(batch_idx)
            loss = loss_fn(train_pred, batch_label)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        true_lable_list = []
        pred_lable_list = []
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for batch_idx, batch_label in test_loader:
            batch_idx = batch_idx.to(DEVICE, dtype=torch.float32)
            batch_label = batch_label.to(DEVICE)
            test_pred = model.forward(batch_idx)
            test_pred_ = torch.argmax(test_pred, dim=1)

            true_lable_list = batch_label.cpu().numpy().tolist()
            pred_lable_list = test_pred_.cpu().numpy().tolist()
            for i in range(len(true_lable_list)):
                if pred_lable_list[i] == 0 and true_lable_list[i] == 0:
                    TP += 1
                if pred_lable_list[i] == 1 and true_lable_list[i] == 1:
                    TN += 1
                if pred_lable_list[i] == 0 and true_lable_list[i] == 1:
                    FP += 1
                if pred_lable_list[i] == 1 and true_lable_list[i] == 0:
                    FN += 1
        accuracy = (TP + TN) * 1.0 / (TP + TN + FP + FN)
        precision = TP * 1.0 / (TP + FP) * 1.0
        recall = TP * 1.0 / (TP + FN)
        f1score = 2.0 * precision * recall / (precision + recall)
        accuracy_list.append(format(accuracy * 100, '.2f'))
        precision_list.append(format(precision * 100, '.2f'))
        recall_list.append(format(recall * 100, '.2f'))
        f1_score_list.append(format(f1score * 100, '.2f'))
        print(accuracy)
        print('---------------------')

    print(accuracy_list)
    print(precision_list)
    print(recall_list)
    print(f1_score_list)
