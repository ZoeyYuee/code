from Config import *
import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader
import torch.nn.functional as F


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
    return texts[:num],labels[:num]


def built_curpus(train_texts, embedding_num):
    word_2_index = {"<PAD>":0, "<UNK>":1}
    for text in train_texts:
        for word in text:
            word_2_index[word] = word_2_index.get(word,len(word_2_index))    # 如果找不到默认值是 len(word_2_index)
    return word_2_index, nn.Embedding(len(word_2_index), embedding_num)   # 随机初始化词向量


class TextDataset(Dataset):
    def __init__(self, all_text, all_label, word_2_index, max_len):
        self.all_text = all_text
        self.all_label = all_label
        self.word_2_index = word_2_index
        self.max_len = max_len

    '''
    1、text：把每篇文章按照长度 200 切分；label 得到该篇文章的标签。
    2、text_idx：得到该文本序列的词向量。
    '''
    def __getitem__(self, index):
        text = self.all_text[index][:self.max_len]
        label = int(self.all_label[index])

        text_idx = [self.word_2_index.get(i, 1) for i in text]    # 没有的用 unk 填充，即 [1]
        text_idx = text_idx + [0] * (self.max_len - len(text_idx))    # 不足 max_len 长度的使用 padding 填充，即 [0]
        text_idx = text_idx[:self.max_len]                      # 大于 max_len 的截取
        text_idx = torch.tensor(text_idx).unsqueeze(dim=0)    # 在第 0 维插入，为了符合 TextCNN 的输入

        return  text_idx, label

    def __len__(self):
        return len(self.all_text)



class TextCNNModel(nn.Module):
    def __init__(self, emb_matrix, max_len, class_num, hidden_num):
        super().__init__()
        self.emb_num = emb_matrix.weight.shape[1]   # embedding 数

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(3, EMBEDDING))
        self.conv2 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(5, EMBEDDING))
        self.conv3 = nn.Conv2d(in_channels=1, out_channels=HIDDEN_DIM, kernel_size=(7, EMBEDDING))

        self.emb_matrix = emb_matrix

        # 作用于最后的全连接层
        self.dropout = nn.Dropout(DROP_PROB)
        self.linear = nn.Linear(hidden_num*3, class_num)  # hidden_num * block_num

    def conv_and_pool(self, conv, input):
        out = conv(input)
        out = F.relu(out)

        return F.max_pool2d(out, (out.shape[2], out.shape[3])).squeeze()

    def forward(self, batch_idx,batch_label=None):
        batch_emb = self.emb_matrix(batch_idx)
        out1 = self.conv_and_pool(self.conv1, batch_emb)
        out2 = self.conv_and_pool(self.conv2, batch_emb)
        out3 = self.conv_and_pool(self.conv3, batch_emb)

        feature = torch.cat([out1, out2, out3], dim=1) # 1* 6 : [ batch * (3 * 2)]

        feature = self.dropout(feature)   # 防止过拟合
        pred = self.linear(feature)   # 全连接层

        return pred



if __name__ == "__main__":
    DEVICE = "cuda:0" if torch.cuda.is_available() else "cpu"  # 判断 gpu 是否可用，不可用就是使用 cpu

    train_text, train_label = read_data(TRAIN_PATH)
    test_text, test_label = read_data(TEST_PATH)

    # words_embedding：词嵌入模型
    word_2_index, words_embedding = built_curpus(train_text, EMBEDDING)

    train_dataset = TextDataset(train_text, train_label, word_2_index, MAX_LEN)
    train_loader = DataLoader(train_dataset, BATH_SIZE, shuffle=True)     # 不随机抓取 200 个样本数据

    test_dataset = TextDataset(test_text, test_label, word_2_index, MAX_LEN)
    test_loader = DataLoader(test_dataset, BATH_SIZE, shuffle=True)


    model = TextCNNModel(words_embedding, MAX_LEN, CLASS_NUM, HIDDEN_DIM).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=LR)
    loss_fn = nn.CrossEntropyLoss()

    accuracy_list = []
    precision_list = []
    recall_list = []
    f1_score_list = []
    for e in range(EPOCH):
        for batch_idx, batch_label in train_loader:
            batch_idx = batch_idx.to(DEVICE)
            batch_label = batch_label.to(DEVICE)
            train_pred = model.forward(batch_idx, batch_label)   # 得到全部特征向量
            loss = loss_fn(train_pred, batch_label)             # 计算损失值

            optimizer.zero_grad()    # 梯度归零
            loss.backward()          # 反向传播
            optimizer.step()         # 参数更新

        true_lable_list = []
        pred_lable_list = []
        TP = 0
        TN = 0
        FP = 0
        FN = 0
        for batch_idx, batch_label in test_loader:
            batch_idx = batch_idx.to(DEVICE)
            batch_label = batch_label.to(DEVICE)
            test_pred = model.forward(batch_idx)
            test_pred_ = torch.argmax(test_pred, dim=1)  # 寻找参数最大值的 idx

            true_lable_list = batch_label.cpu().numpy().tolist()
            pred_lable_list = test_pred_.cpu().numpy().tolist()
            for i in range(len(true_lable_list)):
                if pred_lable_list[i]==0 and true_lable_list[i]==0:
                    TP += 1
                if pred_lable_list[i]==1 and true_lable_list[i]==1:
                    TN += 1
                if pred_lable_list[i]==0 and true_lable_list[i]==1:
                    FP += 1
                if pred_lable_list[i]==1 and true_lable_list[i]==0:
                    FN += 1
        accuracy = (TP + TN) * 1.0 / (TP + TN + FP + FN)
        precision = TP * 1.0 / (TP + FP) * 1.0
        recall = TP * 1.0 / (TP + FN)
        f1_score = 2.0 * precision * recall / (precision + recall)
        accuracy_list.append(format(accuracy * 100, '.2f'))
        precision_list.append(format(precision * 100, '.2f'))
        recall_list.append(format(recall * 100, '.2f'))
        f1_score_list.append(format(f1_score * 100, '.2f'))
        print(accuracy)
        print('---------------------')

    print(accuracy_list)
    print(precision_list)
    print(recall_list)
    print(f1_score_list)
