import torch
import torch.nn as nn
from torch.utils import data
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from Config import *
# Read data function
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

# Custom Dataset class
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
        tokened = self.tokenizer(text, padding='max_length', truncation=True, max_length=MAX_LEN)
        input_ids = tokened['input_ids']
        mask = tokened['attention_mask']

        return torch.tensor(input_ids), torch.tensor(mask), torch.tensor(int(label))

# BERT + TextRNN model
# BERT + TextRNN model
class BERT_TextRNN(nn.Module):
    def __init__(self):
        super(BERT_TextRNN, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        for name, param in self.bert.named_parameters():
            param.requires_grad = False

        self.rnn = nn.LSTM(input_size=EMBEDDING, hidden_size=HIDDEN_DIM, num_layers=N_LAYERS, batch_first=True, bidirectional=True)
        self.rnn.flatten_parameters()
        self.dropout = nn.Dropout(DROP_PROB)
        self.linear = nn.Linear(HIDDEN_DIM * 2, OUTPUT_SIZE)

    def forward(self, input_ids, mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids, attention_mask=mask)[0]
        rnn_output, _ = self.rnn(bert_output)
        rnn_output = rnn_output[:, -1, :]
        output = self.dropout(rnn_output)
        output = self.linear(output)
        return output


if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_dataset = CustomDataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=BATH_SIZE, shuffle=True, drop_last=True)

    test_dataset = CustomDataset('test')
    test_loader = data.DataLoader(test_dataset, batch_size=BATH_SIZE, shuffle=False, drop_last=True)

    model = BERT_TextRNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(1):
        model.train()
        for input_ids, mask, labels in train_loader:
            input_ids, mask, labels = input_ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(input_ids, mask)
            loss = loss_fn(outputs, labels)
            loss.backward()
            optimizer.step()

        model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        with torch.no_grad():
            for input_ids, mask, labels in test_loader:
                input_ids, mask, labels = input_ids.to(DEVICE), mask.to(DEVICE), labels.to(DEVICE)

                outputs = model(input_ids, mask)
                probs = torch.softmax(outputs, dim=1)[:, 1]
                preds = torch.argmax(outputs, dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().detach().numpy())

        accuracy = accuracy_score(all_labels, all_preds)
        precision = precision_score(all_labels, all_preds)
        recall = recall_score(all_labels, all_preds)
        f1 = f1_score(all_labels, all_preds)
        auc_score = roc_auc_score(all_labels, all_probs)

#         print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")
        print(f"Accuracy: {accuracy}")
        print(f"Precision: {precision}")
        print(f"Recall: {recall}")
        print(f"F1 Score: {f1}")
        print(f"AUC: {auc_score}")
        print('-' * 20)
