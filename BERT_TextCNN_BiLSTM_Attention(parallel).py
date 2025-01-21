import sqlite3

import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from  torch.utils.data import Dataset,DataLoader
from Config import *
from torch.utils import data
from transformers import BertTokenizer
from transformers import BertModel
from sklearn.metrics import classification_report, roc_auc_score
import torch
import matplotlib.pyplot as plt

DATABASE = '/kaggle/input/weibo-db/weibo_db.db'


def get_db():
    db = sqlite3.connect(DATABASE)
    return db


def close_connection(db):
    db.close()

def read_data(filename, num=None):
    with open(filename, encoding="utf-8") as f:
        all_data = f.read().split("\n")

    texts = []
    labels = []
    for data in all_data:
        if data:
            t,l = data.split("\t")
            texts.append(t)
            labels.append(l)
    if num == None:
        return texts,labels
    else:
        return texts[:num],labels[:num]


class Dataset(data.Dataset):
    def __init__(self, type='train'):
        global result1
        super().__init__()
        self.conn = sqlite3.connect('/kaggle/input/weibo-db/weibo_db.db')
        self.cursor = self.conn.cursor()

        if type == 'train':
            result1 =self.cursor.execute("SELECT comment,state FROM train_data_seg")
            # self.cursor.execute("SELECT * FROM train_data_seg")
        elif type == 'test':
            result1=self.cursor.execute("SELECT comment,state FROM test_data_seg")

        self.lines=result1.fetchall()

        # self.data = self.cursor.fetchall()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self,index):
        text, label =self.lines[index]
        # print(text)
        # print(label)
        if text is None:
            return self.__getitem__((index + 1) % len(self))

        tokened = self.tokenizer(text)
        input_ids = tokened['input_ids']
        mask = tokened['attention_mask']

        if len(input_ids) < MAX_LEN:
            pad_len = (MAX_LEN - len(input_ids))
            input_ids += [BERT_PAD_ID] * pad_len
            mask += [0] * pad_len

        return torch.tensor(input_ids[:MAX_LEN]), torch.tensor(mask[:MAX_LEN]), torch.tensor(int(label))

    def close(self):
        self.cursor.close()
        self.conn.close()


class BERT_TextCNN_BiLSTM(nn.Module):
    def __init__(self):
        super(BERT_TextCNN_BiLSTM, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        for name, param in self.bert.named_parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(3, EMBEDDING))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(5, EMBEDDING))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(7, EMBEDDING))

        self.lstm = nn.LSTM(input_size=EMBEDDING, hidden_size=HIDDEN_DIM, num_layers=N_LAYERS, batch_first=True, bidirectional=True)

        self.attention = nn.Linear(HIDDEN_DIM*2, 1)

        self.dropout = nn.Dropout(DROP_PROB)
        self.linear = nn.Linear(HIDDEN_DIM*2+HIDDEN_DIM*3, CLASS_NUM)

    def conv_and_pool(self, conv, input):
        out = conv(input)
        out = F.relu(out)
        return F.max_pool2d(out, (out.shape[2], out.shape[3])).squeeze()

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        number = 2
        hidden = (weight.new(N_LAYERS * number, batch_size, HIDDEN_DIM).zero_().float(),
                  weight.new(N_LAYERS * number, batch_size, HIDDEN_DIM).zero_().float()
                  )

        return hidden

    def forward(self, input, mask, hidden):
        bert_out = self.bert(input, mask)[0]  # (batch_size, max_len, hidden_size)

        cnn_out1 = self.conv_and_pool(self.conv1, bert_out.unsqueeze(1))
        cnn_out2 = self.conv_and_pool(self.conv2, bert_out.unsqueeze(1))
        cnn_out3 = self.conv_and_pool(self.conv3, bert_out.unsqueeze(1))
        cnn_out = torch.cat([cnn_out1, cnn_out2, cnn_out3], dim=1)

        lstm_out, (hidden_last, cn_last) = self.lstm(bert_out, hidden)

        lstm_attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        lstm_attention_output = torch.sum(lstm_out * lstm_attention_weights, dim=1)

        all_out = torch.cat([cnn_out, lstm_attention_output], dim=-1)

        all_out = self.dropout(all_out)
        return self.linear(all_out)



if __name__ == "__main__":

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    db = get_db()
    cursor = db.cursor()
    result = cursor.execute("SELECT comment,state FROM train_data_seg").fetchall()
    train_text, train_label = zip(*result)
    test_text, test_label = zip(*cursor.execute("SELECT comment,state FROM test_data_seg").fetchall())

    train_dataset = Dataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=BATH_SIZE, shuffle=True, drop_last=True)

    test_dataset = Dataset('test')
    test_loader = data.DataLoader(test_dataset, batch_size=BATH_SIZE, shuffle=True, drop_last=True)

    model = BERT_TextCNN_BiLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    auc_list = []
    for e in range(5):
        times = 0
        h = model.init_hidden(BATH_SIZE)
        for b, (input, mask, target) in enumerate(train_loader):
            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            h = tuple([each.data for each in h])

            pred = model(input, mask, h)
            loss = loss_fn(pred, target)
            times += 1
            print(f"loss:{loss:.3f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # ------------------  Test  ------------------------

        true_lable_list = []
        pred_lable_list = []
        pred_probability_list = []
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        h = model.init_hidden(BATH_SIZE)
        for b, (test_input, test_mask, test_target) in enumerate(test_loader):
            test_input = test_input.to(DEVICE)
            test_mask = test_mask.to(DEVICE)
            test_target = test_target.to(DEVICE)

            h = tuple([each.data for each in h])

            test_pred = model(test_input, test_mask, h)
            test_pred_ = torch.argmax(test_pred, dim=1)
            true_lable_list = test_target.cpu().numpy().tolist()
            pred_lable_list = test_pred_.cpu().numpy().tolist()
            pred_probability_list = torch.softmax(test_pred, dim=1)[:, 1].detach().cpu().numpy().tolist()

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
        f1_score = 2.0 * precision * recall / (precision + recall)
        # Calculate AUC
        auc = roc_auc_score(true_lable_list, pred_probability_list)
        accuracy_list.append(format(accuracy * 100, '.2f'))
        precision_list.append(format(precision * 100, '.2f'))
        recall_list.append(format(recall * 100, '.2f'))
        f1_score_list.append(format(f1_score * 100, '.2f'))
        auc_list.append(format(auc * 100, '.3f'))
        print(accuracy)
        print(precision)
        print(recall)
        print(f1_score)
        print(auc)
        print('---------------------')

    print(accuracy_list)
    print(precision_list)
    print(recall_list)
    print(f1_score_list)
    print(auc_list)
