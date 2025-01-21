import sqlite3
import jieba
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from sklearn import svm
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from keras.src.utils import pad_sequences
from keras.preprocessing.text import Tokenizer
from sklearnex import patch_sklearn
patch_sklearn()

DATABASE = 'weibo_db.db'

def get_db():
    db = sqlite3.connect(DATABASE)
    return db

def close_connection(db):
    db.close()

def predict_test_data(user):
    db = get_db()
    cursor = db.cursor()

    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])

    word2vec_model = Word2Vec.load("word2vec_model.bin")

    df['comment'] = df['comment'].apply(lambda x: jieba.lcut(str(x)))

    vectors = []
    for comment in df['comment']:
        vec = np.zeros(word2vec_model.vector_size)
        count = 0
        for word in comment:
            if word in word2vec_model.wv:
                vec += word2vec_model.wv[word]
                count += 1
        if count != 0:
            vec /= count
        vectors.append(vec)

    X_train = np.array(vectors)
    y_train = np.array(df['state'].values.tolist())

    model = svm.SVC(C=1, kernel='rbf', random_state=42)
    model.fit(X_train, y_train)

    db = get_db()
    cursor = db.cursor()

    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])
    df.dropna(inplace=True)

    df['comment'] = df['comment'].apply(lambda x: jieba.lcut(str(x)))

    vectors = []
    for comment in df['comment']:
        vec = np.zeros(word2vec_model.vector_size)
        count = 0
        for word in comment:
            if word in word2vec_model.wv:
                vec += word2vec_model.wv[word]
                count += 1
        if count != 0:
            vec /= count
        vectors.append(vec)

    X_test = np.array(vectors)
    y_test = np.array(df['state'].values.tolist())

    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    predictions = model.predict(data_sequences)
    risk = [p[0] for p in predictions]
    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]

    db = get_db()
    cursor = db.cursor()
    for i, state in enumerate(predicted_states):
        query = "UPDATE test_data_seg SET trisk = ? WHERE id = ?"
        parameters = (float(risk[i]), int(df['id'][i]))
        cursor.execute(query, parameters)
    db.commit()
    close_connection(db)

    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)

    return accuracy, precision, recall, f1

def calculate_metrics(user):
    db = get_db()
    cursor = db.cursor()

    cursor.execute("SELECT weibo_id, state FROM wb_userinfo WHERE user = ?", (user,))
    userinfo_data = cursor.fetchall()

    metrics_data = []
    for userinfo in userinfo_data:
        weibo_id = userinfo[0]
        state = userinfo[1]

        cursor.execute("SELECT trisk FROM test_data_seg WHERE user = ? AND weibo_id = ?", (user, weibo_id))
        trisk_data = cursor.fetchall()
        trisk_data = [float(data[0]) for data in trisk_data]

        trisk_avg = np.mean(trisk_data)

        cursor.execute("UPDATE wb_userinfo SET trisk = ? WHERE user = ? AND weibo_id = ?", (trisk_avg, user, weibo_id))
        metrics_data.append((weibo_id, trisk_avg, state))

    db.commit()
    close_connection(db)

    df = pd.DataFrame(metrics_data, columns=['weibo_id', 'trisk_avg', 'state'])

    accuracy = accuracy_score(df['state'], df['trisk_avg'].round())
    precision = precision_score(df['state'], df['trisk_avg'].round())
    recall = recall_score(df['state'], df['trisk_avg'].round())
    f1 = f1_score(df['state'], df['trisk_avg'].round())

    return float(accuracy), float(precision), float(recall), float(f1)

predict_test_data("2")
