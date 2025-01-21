import sqlite3

from transformers import BertTokenizer, BertModel
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import torch
from Config import *


DATABASE = 'weibo_db.db'

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
            t, l = data.split("\t")
            texts.append(t)
            labels.append(int(l))  # Convert label to integer
    if num == None:
        return texts, labels
    else:
        return texts[:num], labels[:num]

if __name__ == "__main__":

    tokenizer = BertTokenizer.from_pretrained(BERT_MODEL)
    model = BertModel.from_pretrained(BERT_MODEL)
    model.eval()

    db = get_db()
    cursor = db.cursor()
    result = cursor.execute("SELECT comment,state FROM train_data_seg").fetchall()
    train_text, train_label = zip(*result)
    test_text, test_label = zip(*cursor.execute("SELECT comment,state FROM test_data_seg").fetchall())

    train_embeddings = []
    for text in train_text:
        if text is None:
            continue
        tokenized_text = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN)
        with torch.no_grad():
            outputs = model(**tokenized_text)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        train_embeddings.append(embeddings)

    test_embeddings = []
    for text in test_text:
        if text is None:
            continue
        tokenized_text = tokenizer(text, return_tensors='pt', padding='max_length', truncation=True, max_length=MAX_LEN)
        with torch.no_grad():
            outputs = model(**tokenized_text)
        embeddings = outputs.last_hidden_state.mean(dim=1).squeeze().cpu().numpy()
        test_embeddings.append(embeddings)

    train_embeddings = torch.tensor(train_embeddings)
    test_embeddings = torch.tensor(test_embeddings)

    clf = RandomForestClassifier(n_estimators=100)
    clf.fit(train_embeddings, train_label)

    pred_labels = clf.predict(test_embeddings)
    pred_probs = clf.predict_proba(test_embeddings)[:, 1]  # Probability for positive class

    accuracy = accuracy_score(test_label, pred_labels)
    precision = precision_score(test_label, pred_labels, average='macro')
    recall = recall_score(test_label, pred_labels, average='macro')
    f1 = f1_score(test_label, pred_labels, average='macro')
    auc = roc_auc_score(test_label, pred_probs)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)
    print("AUC:", auc)
