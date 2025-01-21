import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.cluster.hierarchy import weighted
from torch.utils.data import Dataset, DataLoader
from Config import *
from torch.utils import data
from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report, roc_auc_score, roc_curve, confusion_matrix
import torch
import sqlite3
import json

DATABASE = '/kaggle/working/weibo_db.db'

def save_auc_to_db():
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    create_table_sql = '''
CREATE TABLE IF NOT EXISTS fpr_tpr_information_gain_auc (
    name TEXT PRIMARY KEY,
    fpr TEXT NOT NULL,
    tpr TEXT NOT NULL
);
'''
    cursor.execute(create_table_sql)
    conn.commit()
    conn.close()

def insert_to_auc_db(name, fpr, tpr):
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    insert_sql = '''
INSERT OR REPLACE INTO fpr_tpr_information_gain_auc (name, fpr, tpr) VALUES (?, ?, ?);
'''
    cursor.execute(insert_sql, (name, json.dumps(fpr), json.dumps(tpr)))
    conn.commit()
    conn.close()

def read_data(filename, num=None):
    with open(filename, encoding="utf-8") as f:
        all_data = f.read().split("\n")

    texts = []
    labels = []
    for data in all_data:
        if data:
            t, l = data.split("\t")
            texts.append(t)
            labels.append(int(l))
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
        tokened = self.tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LEN, return_tensors='pt')
        input_ids = tokened['input_ids'].squeeze()
        mask = tokened['attention_mask'].squeeze()

        return input_ids, mask, torch.tensor(int(label))

def compute_entropy(probabilities):
    eps = 1e-12
    entropy = -np.sum(probabilities * np.log2(probabilities + eps), axis=1)
    return entropy

class BERT_TextCNN_BiLSTM_IG(nn.Module):
    def __init__(self):
        super(BERT_TextCNN_BiLSTM_IG, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        for name, param in self.bert.named_parameters():
            param.requires_grad = False  # Freeze BERT parameters

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(3, 300))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(5, 300))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(7, 300))

        self.lstm = nn.LSTM(input_size=HIDDEN_DIM*3, hidden_size=HIDDEN_DIM, num_layers=N_LAYERS, batch_first=True, bidirectional=True)

        self.attention = nn.Linear(HIDDEN_DIM*2, 1)  # Attention layer to calculate attention weights

        self.dropout = nn.Dropout(DROP_PROB)

        self.info_gain = nn.Linear(HIDDEN_DIM*2, 1)
        self.linear = nn.Linear(HIDDEN_DIM*2, CLASS_NUM)

    def conv_and_relu(self, conv, input):
        out = conv(input)
        out = F.relu(out)
        return out

    def init_hidden(self, batch_size):
        weight = next(self.parameters()).data
        number = 2
        hidden = (weight.new(N_LAYERS * number, batch_size, HIDDEN_DIM).zero_().float().to(DEVICE),
                  weight.new(N_LAYERS * number, batch_size, HIDDEN_DIM).zero_().float().to(DEVICE))
        return hidden

    def attention_fusion(self, lstm_out):
        attention_weights = F.softmax(self.attention(lstm_out), dim=1)
        weighted_lstm_out = lstm_out * attention_weights
        attended_features = torch.sum(weighted_lstm_out, dim=1)
        return attended_features

    def forward(self, input_ids, mask, hidden):
        bert_out = self.bert(input_ids=input_ids, attention_mask=mask)[0]

        cnn_out1 = self.conv_and_relu(self.conv1, bert_out.unsqueeze(1))
        cnn_out2 = self.conv_and_relu(self.conv2, bert_out.unsqueeze(1))
        cnn_out3 = self.conv_and_relu(self.conv3, bert_out.unsqueeze(1))

        cnn_out1 = cnn_out1.squeeze(-1)
        cnn_out2 = cnn_out2.squeeze(-1)
        cnn_out3 = cnn_out3.squeeze(-1)

        cnn_out = torch.cat([cnn_out1, cnn_out2, cnn_out3], dim=1)
        cnn_out = cnn_out.permute(0, 2, 1)

        lstm_out, (hidden_last, cn_last) = self.lstm(cnn_out, hidden)

        ig_weights = torch.sigmoid(self.info_gain(lstm_out))
        weighted_lstm_out = lstm_out * ig_weights

        attended_features = self.attention_fusion(weighted_lstm_out)

        all_out = self.dropout(attended_features)

        logits = self.linear(all_out)

        return logits

if __name__ == "__main__":
    save_auc_to_db()
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_text, train_label = read_data(TRAIN_PATH)
    test_text, test_label = read_data(TEST_PATH)

    train_dataset = CustomDataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)

    test_dataset = CustomDataset('test')
    test_loader = data.DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

    model = BERT_TextCNN_BiLSTM_IG().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    auc_list = []
    entropy_list = []

    for e in range(EPOCH):
        model.train()
        for b, (input_ids, mask, target) in enumerate(train_loader):
            input_ids = input_ids.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)


            h = model.init_hidden(input_ids.size(0))

            pred = model(input_ids, mask, h)
            loss = loss_fn(pred, target)
            print(f"Epoch {e+1}, Batch {b+1}, Loss: {loss.item():.3f}")

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()


        model.eval()
        true_label_list = []
        pred_label_list = []
        pred_prob_list = []
        all_probs = []

        with torch.no_grad():
            for b, (test_input, test_mask, test_target) in enumerate(test_loader):
                test_input = test_input.to(DEVICE)
                test_mask = test_mask.to(DEVICE)
                test_target = test_target.to(DEVICE)

                # 根据当前batch大小初始化隐藏状态
                h = model.init_hidden(test_input.size(0))

                test_pred = model(test_input, test_mask, h)
                test_pred_ = torch.argmax(test_pred, dim=1)
                test_pred_prob = F.softmax(test_pred, dim=1)[:, 1]

                true_label_list.extend(test_target.cpu().numpy().tolist())
                pred_label_list.extend(test_pred_.cpu().numpy().tolist())
                pred_prob_list.extend(test_pred_prob.cpu().numpy().tolist())


                all_probs.append(F.softmax(test_pred, dim=1).cpu().numpy())


        all_probs = np.vstack(all_probs)


        cm = confusion_matrix(true_label_list, pred_label_list)
        TP, TN, FP, FN = cm[1, 1], cm[0, 0], cm[0, 1], cm[1, 0]

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        auc = roc_auc_score(true_label_list, pred_prob_list) if len(set(true_label_list)) > 1 else 0.5


        entropies = compute_entropy(all_probs)
        normalized_entropy = np.mean(entropies)


        accuracy_list.append(round(accuracy * 100, 2))
        precision_list.append(round(precision * 100, 2))
        recall_list.append(round(recall * 100, 2))
        f1_score_list.append(round(f1_score * 100, 2))
        auc_list.append(round(auc * 100, 2))
        entropy_list.append(round(normalized_entropy, 4))


        fpr, tpr, _ = roc_curve(true_label_list, pred_prob_list)
        insert_to_auc_db(f'epoch_{e+1}', fpr.tolist(), tpr.tolist())


        print(f"Epoch {e+1} :")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Normalized Entropy: {normalized_entropy:.4f}")
        print('---------------------')


    print("Accuracy List:", accuracy_list)
    print("Precision List:", precision_list)
    print("Recall List:", recall_list)
    print("F1 Score List:", f1_score_list)
    print("AUC List:", auc_list)
    print("Entropy List:", entropy_list)
