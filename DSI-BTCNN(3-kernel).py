import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
from Config import *
from torch.utils import data
from transformers import BertTokenizer, BertModel
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
import torch
import matplotlib.pyplot as plt
import sqlite3

DATABASE = '/kaggle/working/weibo_db.db'


def compute_entropy(probabilities):
    eps = 1e-12  # 防止对数计算时出现零值
    entropy = -np.sum(probabilities * np.log2(probabilities + eps), axis=1)
    return entropy


def insert_to_auc_db(name, fpr, tpr):
    """
    Insert FPR and TPR values into the SQLite database.
    Serializes the arrays as strings for storage.
    """
    # Connect to SQLite database
    conn = sqlite3.connect(DATABASE)
    cursor = conn.cursor()
    insert_sql = '''
    INSERT OR REPLACE INTO fpr_tpr_mulpti_textcnn_auc (name, fpr, tpr) VALUES (?, ?, ?);
    '''
    # Serialize fpr and tpr as strings
    fpr_str = ','.join(map(str, fpr))
    tpr_str = ','.join(map(str, tpr))
    cursor.execute(insert_sql, (name, fpr_str, tpr_str))
    conn.commit()
    conn.close()


def read_data(filename, num=None):
    """
    Read data from a file. Each line should contain text and label separated by a tab.
    """
    with open(filename, encoding="utf-8") as f:
        all_data = f.read().split("\n")

    texts = []
    labels = []
    for data in all_data:
        if data:
            parts = data.split("\t")
            if len(parts) != 2:
                continue  # Skip malformed lines
            t, l = parts
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
        else:
            raise ValueError("Dataset type must be 'train' or 'test'.")

        self.lines = open(sample_path, encoding='utf-8').readlines()
        self.tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)

    def __len__(self):
        return len(self.lines)

    def __getitem__(self, index):
        parts = self.lines[index].strip().split('\t')
        if len(parts) != 2:
            raise ValueError(f"Line {index} is malformed: {self.lines[index]}")
        text, label = parts
        tokened = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=MAX_LEN,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        input_ids = tokened['input_ids'].squeeze()
        mask = tokened['attention_mask'].squeeze()

        return input_ids, mask, torch.tensor(int(label))


class Attention(nn.Module):
    def __init__(self, hidden_dim):
        super(Attention, self).__init__()
        self.attention = nn.Linear(hidden_dim, 1)

    def forward(self, x):
        """
        x: Tensor of shape (batch_size, hidden_dim)
        """
        scores = self.attention(x)  # (batch_size, 1)
        attention_weights = torch.sigmoid(scores)  # (batch_size, 1)
        context_vector = x * attention_weights  # (batch_size, hidden_dim)
        return context_vector


class BERT_TextCNN_BiLSTM(nn.Module):
    def __init__(self):
        super(BERT_TextCNN_BiLSTM, self).__init__()

        # Initialize BERT model
        self.bert = BertModel.from_pretrained(BERT_MODEL)
        for name, param in self.bert.named_parameters():
            param.requires_grad = False  # Freeze BERT parameters

        # Define convolutional layers with different kernel sizes
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(3, EMBEDDING))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(5, EMBEDDING))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(7, EMBEDDING))
       #self.conv4 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(9, EMBEDDING))四核
        # Define BiLSTM layer
        self.lstm = nn.LSTM(
            input_size=EMBEDDING,
            hidden_size=HIDDEN_DIM,
            num_layers=N_LAYERS,
            batch_first=True,
            bidirectional=True
        )

        # Define dropout layer
        self.dropout = nn.Dropout(DROP_PROB)

        # Define linear layers to project CNN and LSTM outputs to the same dimension
        self.cnn_linear = nn.Linear(HIDDEN_DIM * 3, HIDDEN_DIM) #4核改成×4
        self.lstm_linear = nn.Linear(HIDDEN_DIM * 2, HIDDEN_DIM)

        # Define attention layer after fusion
        self.attention_fusion = Attention(HIDDEN_DIM)

        # Define fully connected layer
        self.linear = nn.Linear(HIDDEN_DIM, CLASS_NUM)

    def conv_and_pool(self, conv_layer, input_tensor):
        """
        Apply convolution and max pooling to the input tensor.
        """
        out = conv_layer(input_tensor)  # (batch, out_channels, H, W)
        out = F.relu(out)
        # Max pooling over the entire feature map
        out = F.max_pool2d(out, (out.shape[2], out.shape[3])).squeeze()
        return out  # (batch, out_channels)

    def init_hidden(self, batch_size):
        """
        Initialize hidden states for LSTM.
        """
        weight = next(self.parameters()).data
        num_directions = 2  # Because bidirectional
        hidden = (
            weight.new(N_LAYERS * num_directions, batch_size, HIDDEN_DIM).zero_().float(),
            weight.new(N_LAYERS * num_directions, batch_size, HIDDEN_DIM).zero_().float()
        )
        return hidden

    def information_gain(self, x):
        """
        Calculate information gain for the input tensor.
        """
        p_x_pos = torch.mean(x[x > 0]) if (x[x > 0].numel() > 0) else 0.01
        p_x_neg = torch.mean(x[x <= 0]) if (x[x <= 0].numel() > 0) else 0.01
        ig_value_x_pos = -p_x_pos * torch.log2(p_x_pos + 1e-12)
        ig_value_x_neg = -p_x_neg * torch.log2(p_x_neg + 1e-12)
        return ig_value_x_pos + ig_value_x_neg

    def forward(self, input_tensor, mask, hidden):
        """
        Forward pass through the network.
        """
        # BERT embeddings
        bert_output = self.bert(input_tensor, attention_mask=mask)
        sequence_output = bert_output.last_hidden_state  # (batch, seq_len, hidden_dim)

        # For CNN
        bert_cnn_out = sequence_output.unsqueeze(1)  # (batch, 1, seq_len, hidden_dim)

        # Apply convolution and pooling
        cnn_out1 = self.conv_and_pool(self.conv1, bert_cnn_out)  # (batch, HIDDEN_DIM)
        cnn_out2 = self.conv_and_pool(self.conv2, bert_cnn_out)  # (batch, HIDDEN_DIM)
        cnn_out3 = self.conv_and_pool(self.conv3, bert_cnn_out)  # (batch, HIDDEN_DIM)
        cnn_out = torch.cat([cnn_out1, cnn_out2, cnn_out3], dim=1)  # (batch, HIDDEN_DIM*3)


        # Project CNN output to HIDDEN_DIM
        cnn_proj = self.cnn_linear(cnn_out)  # (batch, HIDDEN_DIM)

        # For LSTM
        lstm_out, (hidden_last, cn_last) = self.lstm(sequence_output, hidden)  # lstm_out: (batch, seq_len, HIDDEN_DIM*2)

        # Mean pooling for LSTM output
        lstm_mean = torch.mean(lstm_out, dim=1)  # (batch, HIDDEN_DIM*2)

        # Project LSTM output to HIDDEN_DIM
        lstm_proj = self.lstm_linear(lstm_mean)  # (batch, HIDDEN_DIM)

        # Calculate information gain
        info_gain_cnn = self.information_gain(cnn_proj)  # Scalar
        info_gain_lstm = self.information_gain(lstm_proj)  # Scalar
        total_info_gain = info_gain_cnn + info_gain_lstm

        # Avoid division by zero
        weight_cnn = (info_gain_cnn / total_info_gain) if total_info_gain > 0 else 0.5
        weight_lstm = (info_gain_lstm / total_info_gain) if total_info_gain > 0 else 0.5

        # Feature fusion using information gain
        fused_features = weight_cnn * cnn_proj + weight_lstm * lstm_proj  # (batch, HIDDEN_DIM)

        # Apply attention on fused features
        attended_features = self.attention_fusion(fused_features)  # (batch, HIDDEN_DIM)

        # Apply dropout
        all_out = self.dropout(attended_features)  # (batch, HIDDEN_DIM)

        # Fully connected layer
        logits = self.linear(all_out)  # (batch, CLASS_NUM)

        return logits


if __name__ == "__main__":
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Read data
    train_text, train_label = read_data(TRAIN_PATH)
    test_text, test_label = read_data(TEST_PATH)

    # Create datasets and dataloaders
    train_dataset = CustomDataset('train')
    train_loader = data.DataLoader(
        train_dataset,
        batch_size=BATH_SIZE,
        shuffle=True,
        drop_last=True
    )

    test_dataset = CustomDataset('test')
    test_loader = data.DataLoader(
        test_dataset,
        batch_size=BATH_SIZE,
        shuffle=False,  # Typically, shuffle=False for evaluation
        drop_last=False
    )

    # Initialize model, optimizer, loss function
    model = BERT_TextCNN_BiLSTM().to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    # Metrics lists
    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    auc_list = []

    num_epochs = 5  # Increased epochs for better training

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        model.train()
        h = model.init_hidden(BATH_SIZE)
        for b, (input_ids, mask, target) in enumerate(train_loader):
            input_ids = input_ids.to(DEVICE)
            mask = mask.to(DEVICE)
            target = target.to(DEVICE)

            # Detach hidden states to prevent backpropagating through entire training history
            h = tuple([each.detach() for each in h])

            # Forward pass
            pred = model(input_ids, mask, h)
            loss = loss_fn(pred, target)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()

            if (b + 1) % 10 == 0 or (b + 1) == len(train_loader):
                print(f"Batch {b + 1}/{len(train_loader)} - Loss: {loss.item():.4f}")

        # Evaluation
        model.eval()
        true_label_list = []
        pred_label_list = []
        pred_prob_list = []
        all_probs = []
        TP = TN = FP = FN = 0

        with torch.no_grad():
            h = model.init_hidden(BATH_SIZE)
            for b, (test_input, test_mask, test_target) in enumerate(test_loader):
                test_input = test_input.to(DEVICE)
                test_mask = test_mask.to(DEVICE)
                test_target = test_target.to(DEVICE)

                h = tuple([each.detach() for each in h])

                test_pred = model(test_input, test_mask, h)
                test_pred_ = torch.argmax(test_pred, dim=1)

                # Get probabilities
                test_pred_prob_all = F.softmax(test_pred, dim=1)  # (batch, CLASS_NUM)
                test_pred_prob = test_pred_prob_all  # Probability of the positive class

                probs = test_pred_prob.cpu().detach().numpy()  # 转为 NumPy 数组
                all_probs.append(probs)
                # Save all probabilities

                true_label_list.extend(test_target.cpu().numpy().tolist())
                pred_label_list.extend(test_pred_.cpu().numpy().tolist())
                pred_prob_list.extend(test_pred_prob[:, 1].cpu().numpy().tolist())

    # 合并所有概率
        all_probs = np.vstack(all_probs)

        # 计算混淆矩阵
        for i in range(len(true_label_list)):
            if pred_label_list[i] == 1 and true_label_list[i] == 1:
                TP += 1
            elif pred_label_list[i] == 0 and true_label_list[i] == 0:
                TN += 1
            elif pred_label_list[i] == 1 and true_label_list[i] == 0:
                FP += 1
            elif pred_label_list[i] == 0 and true_label_list[i] == 1:
                FN += 1

        accuracy = (TP + TN) / (TP + TN + FP + FN) if (TP + TN + FP + FN) > 0 else 0
        precision = TP / (TP + FP) if (TP + FP) != 0 else 0
        recall = TP / (TP + FN) if (TP + FN) != 0 else 0
        f1_score = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        auc = roc_auc_score(true_label_list, pred_prob_list) if len(set(true_label_list)) > 1 else 0.5

        # 计算归一化熵

        entropies = compute_entropy(all_probs)
        normalized_entropy = np.mean(entropies)

        accuracy_list.append(format(accuracy * 100, '.2f'))
        precision_list.append(format(precision * 100, '.2f'))
        recall_list.append(format(recall * 100, '.2f'))
        f1_score_list.append(format(f1_score * 100, '.2f'))
        auc_list.append(format(auc * 100, '.2f'))

        print(f"Accuracy: {accuracy:.4f}")
        print(f"AUC: {auc:.4f}")
        print(f"Normalized Entropy: {normalized_entropy:.4f}")
        print('---------------------')

    # 输出所有评估指标
    print("Accuracy List:", accuracy_list)
    print("Precision List:", precision_list)
    print("Recall List:", recall_list)
    print("F1 Score List:", f1_score_list)
    print("AUC List:", auc_list)