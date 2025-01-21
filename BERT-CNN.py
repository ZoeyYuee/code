import torch
import torch.nn as nn
from torch.utils import data
from transformers import BertTokenizer, BertModel
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from Config import *


# Configuration constants
# TRAIN_PATH = "/kaggle/input/train-test-data/data/after.txt"
# TEST_PATH = "/kaggle/input/train-test-data/data/dev.txt"
# BERT_MODEL = "/kaggle/input/bert-base-chinese/bert-base-chinese"
# MAX_LEN = 128
# BATCH_SIZE = 32
# EMBEDDING = 768  # BERT base model embedding size
# NUM_FILTERS = 100
# KERNEL_SIZE = 3
# OUTPUT_SIZE = 2  # For binary classification
# DROP_PROB = 0.5
# LR = 1e-4
# NUM_EPOCHS = 3

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


# BERT + CNN model
class BERT_CNN(nn.Module):
    def __init__(self):
        super(BERT_CNN, self).__init__()
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        for name, param in self.bert.named_parameters():
            param.requires_grad = False

        self.conv1 = nn.Conv1d(in_channels=EMBEDDING, out_channels=HIDDEN_DIM, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=HIDDEN_DIM, out_channels=HIDDEN_DIM, kernel_size=3, padding=1)
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        self.dropout = nn.Dropout(DROP_PROB)
        self.linear = nn.Linear(HIDDEN_DIM * (MAX_LEN // 2 // 2), OUTPUT_SIZE)

    def forward(self, input_ids, mask):
        with torch.no_grad():
            bert_output = self.bert(input_ids, attention_mask=mask)[0]

        # Transpose to match Conv1d input shape (batch_size, embed_dim, seq_len)
        bert_output = bert_output.transpose(1, 2)

        x = self.conv1(bert_output)
        x = F.relu(x)
        x = self.pool(x)

        x = self.conv2(x)
        x = F.relu(x)
        x = self.pool(x)

        # Flatten the output for the linear layer
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        output = self.linear(x)
        return output


if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    train_text, train_label = read_data(TRAIN_PATH)
    test_text, test_label = read_data(TEST_PATH)

    train_dataset = CustomDataset('train')
    train_loader = data.DataLoader(train_dataset, batch_size=BATH_SIZE, shuffle=True, drop_last=True)

    test_dataset = CustomDataset('test')
    test_loader = data.DataLoader(test_dataset, batch_size=BATH_SIZE, shuffle=True, drop_last=True)

    model = BERT_CNN().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    auc_list = []

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

        true_label_list = []
        pred_label_list = []
        pred_prob_list = []
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

            for i in range(len(true_label_list)):
                if pred_label_list[i] == 0 and true_label_list[i] == 0:
                    TP += 1
                if pred_label_list[i] == 1 and true_label_list[i] == 1:
                    TN += 1
                if pred_label_list[i] == 0 and true_label_list[i] == 1:
                    FP += 1
                if pred_label_list[i] == 1 and true_label_list[i] == 0:
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

        print(f"Accuracy: {accuracy}")
        print(f"AUC: {auc}")
        print('---------------------')

    print("BERT_CNN")
    print(accuracy_list)
    print(precision_list)
    print(recall_list)
    print(f1_score_list)
    print(auc_list)

