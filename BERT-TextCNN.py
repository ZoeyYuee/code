import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Config import *
from torch.utils import data
from transformers import BertTokenizer
from transformers import BertModel
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import torch
import matplotlib.pyplot as plt

def read_data(filename, num=None):
    with open(filename, encoding="utf-8") as f:
        all_data = f.read().split("\n")

    texts = []
    labels = []
    for data in all_data:
        if data:
            t, l = data.split("\t")
            texts.append(t)
            labels.append(l)
    if num is None:
        return texts, labels
    else:
        return texts[:num], labels[:num]


class CustomDataset(data.Dataset):
    def __init__(self, type='train'):
        super().__init__()
        if type == 'train':
            sample_path = TRAIN_PATH
        elif type == 'test':
            sample_path = TEST_PATH

        self.lines = open(sample_path, encoding='utf-8').readlines()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        text, label = self.lines[index].split('\t')
        tokened = self.tokenizer(text)
        input_ids = tokened['input_ids']
        mask = tokened['attention_mask']

        if len(input_ids) < MAX_LEN:
            pad_len = (MAX_LEN - len(input_ids))
            input_ids += [BERT_PAD_ID] * pad_len
            mask += [0] * pad_len

        return torch.tensor(input_ids[:MAX_LEN]), torch.tensor(mask[:MAX_LEN]), torch.tensor(int(label))


class BERT_TextCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        for name, param in self.bert.named_parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(3, EMBEDDING))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(5, EMBEDDING))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(7, EMBEDDING))

        self.dropout = nn.Dropout(DROP_PROB)
        self.linear = nn.Linear(HIDDEN_DIM*3, CLASS_NUM)

    def conv_and_pool(self, conv, input):
        out = conv(input)
        out = F.relu(out)

        # out.shape[2]: MAX_LEN - kernel_num + 1
        # out.shape[3]: 1

        return F.max_pool2d(out, (out.shape[2], out.shape[3])).squeeze()

    def forward(self, input, mask):
        out = self.bert(input, mask)[0].unsqueeze(1)
        out1 = self.conv_and_pool(self.conv1, out)
        out2 = self.conv_and_pool(self.conv2, out)
        out3 = self.conv_and_pool(self.conv3, out)

        out = torch.cat([out1, out2, out3], dim=1)

        out = self.dropout(out)

        return self.linear(out)


if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_text, train_label = read_data(TRAIN_PATH)
    test_text, test_label = read_data(TEST_PATH)

    train_dataset = CustomDataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=BATH_SIZE, shuffle=True)

    test_dataset =CustomDataset('test')
    test_loader = data.DataLoader(test_dataset, batch_size=BATH_SIZE, shuffle=True)

    model = BERT_TextCNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    for e in range(1):
        times = 0
        for b, (input, mask, target) in enumerate(train_loader):
            input = input.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            pred = model(input, mask)
            loss = loss_fn(pred, target)
            times += 1
            print(f"loss:{loss:.3f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        # ------------------  Test  ------------------------

        true_lable_list = []
        pred_lable_list = []
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for b, (test_input, test_mask, test_target) in enumerate(test_loader):
            test_input = test_input.to(DEVICE)
            test_mask = test_mask.to(DEVICE)
            test_target = test_target.to(DEVICE)

            test_pred = model(test_input, test_mask)
            test_pred_ = torch.argmax(test_pred, dim=1)
            test_pred_prob = F.softmax(test_pred, dim=1)[:, 1]

            true_label_list.extend(test_target.cpu().numpy().tolist())
            pred_label_list.extend(test_pred_.cpu().numpy().tolist())
            pred_prob_list.extend(test_pred_prob.cpu().detach().numpy().tolist())

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
        auc = roc_auc_score(true_label_list, pred_prob_list)

        accuracy_list.append(format(accuracy * 100, '.2f'))
        precision_list.append(format(precision * 100, '.2f'))
        recall_list.append(format(recall * 100, '.2f'))
        f1_score_list.append(format(f1_score * 100, '.2f'))
        auc_list.append(format(auc * 100, '.2f'))
        print(accuracy)
        print('---------------------')

    print('BERT-TextCNN')
    print(accuracy_list)
    print(precision_list)
    print(recall_list)
    print(f1_score_list)
    print(auc_list)
