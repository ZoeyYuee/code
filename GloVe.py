import random
import sqlite3
import threading
from decimal import getcontext, Decimal

import gensim
import jieba
import networkx as nx
import numpy as np
import pandas as pd
from gensim.models import Word2Vec
from keras import Input, Model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPooling1D, SimpleRNN, Bidirectional, GRU, Flatten, \
    concatenate
from keras.callbacks import ModelCheckpoint, Callback
from keras.preprocessing.text import Tokenizer
from matplotlib import pyplot as plt
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, \
    mean_absolute_error, mean_squared_error, matthews_corrcoef, roc_curve, auc
from keras.models import load_model
from torch.optim import Adam

DATABASE = 'weibo_db.db'

def get_db():
    db = sqlite3.connect(DATABASE)
    return db


def close_connection(db):
    db.close()

def load_glove_embedding(file_path):
    embeddings_index = {}
    with open(file_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            coefs = np.asarray(values[1:], dtype='float32')
            embeddings_index[word] = coefs
    return embeddings_index

def train_text_cnn(user):


    db = get_db()
    cursor = db.cursor()


    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()


    close_connection(db)


    df = pd.DataFrame(result, columns=['comment', 'state'])
    # df = df.sample(frac=0.8, random_state=42)


    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")


    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))



    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())


    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)


    max_sequence_length = 180


    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)


    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])
    num_words = min(len(word_index) + 1, len(glove_model))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(Conv1D(128, 2, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 4, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])



    checkpoint = ModelCheckpoint('glove_text_cnn_model.h5', monitor='val_accuracy', mode='max', save_best_only=True,
                                 verbose=1)


    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("Epoch:", epoch + 1)
            print("Train loss:", logs['loss'])
            print("Train accuracy:", logs['accuracy'])
            print("Test loss:", logs['val_loss'])


    loss_history = LossHistory()


    history = model.fit(data_sequences, labels, validation_split=0.2, epochs=5, batch_size=32, callbacks=[checkpoint, loss_history])

    return history





def predict_test_data(user):
    db = get_db()
    cursor = db.cursor()


    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()


    close_connection(db)


    df = pd.DataFrame(result, columns=['comment', 'state'])


    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")


    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))


    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())


    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)


    max_sequence_length = 180


    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)


    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])
    num_words = min(len(word_index) + 1, len(glove_model))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector


    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(Conv1D(128, 2, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 4, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights('glove_text_cnn_model.h5')

    db = get_db()
    cursor = db.cursor()


    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()


    close_connection(db)


    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])


    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))


    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())


    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    predictions = model.predict(data_sequences)

    db = get_db()
    cursor = db.cursor()


    for i, p in enumerate(predictions):

        record_id = df.loc[i, 'id']
        cursor.execute("UPDATE test_data_seg SET trisk = ? WHERE id = ?", (float(p), str(record_id)))
        db.commit()



    db.commit()
    close_connection(db)

    # risk = [p[0] for p in predictions]
    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]

    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)
    auc = roc_auc_score(states, predictions)

    return accuracy, precision, recall, f1, auc

# train_text_cnn("2")
# print(predict_test_data("2"))


def train_rnn(user):
    db = get_db()
    cursor = db.cursor()

    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])

    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])
    num_words = min(len(word_index) + 1, len(glove_model))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(SimpleRNN(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('glove_rnn_model.h5', monitor='val_accuracy', mode='max', save_best_only=True,
                                 verbose=1)

    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("Epoch:", epoch + 1)
            print("Train loss:", logs['loss'])
            print("Train accuracy:", logs['accuracy'])
            print("Test loss:", logs['val_loss'])

    loss_history = LossHistory()

    history = model.fit(data_sequences, labels, validation_split=0.2, epochs=5, batch_size=32, callbacks=[checkpoint, loss_history])

    return history

def predict_test_data_rnn(user):
    db = get_db()
    cursor = db.cursor()

    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])

    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])
    num_words = min(len(word_index) + 1, len(glove_model))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(
        Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                  trainable=False))
    model.add(SimpleRNN(128, activation='relu'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights('glove_rnn_model.h5')

    db = get_db()
    cursor = db.cursor()

    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())

    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    predictions = model.predict(data_sequences)

    db = get_db()
    cursor = db.cursor()

    for i, p in enumerate(predictions):
        record_id = df.loc[i, 'id']
        cursor.execute("UPDATE test_data_seg SET trisk = ? WHERE id = ?", (float(p), str(record_id)))
        db.commit()

    db.commit()
    close_connection(db)

    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]

    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)
    auc = roc_auc_score(states, predictions)

    return accuracy, precision, recall, f1, auc


from keras.layers import LSTM

def train_lstm(user):
    db = get_db()
    cursor = db.cursor()

    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])

    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])
    num_words = min(len(word_index) + 1, len(glove_model))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('glove_lstm_model.h5', monitor='val_accuracy', mode='max', save_best_only=True,
                                 verbose=1)

    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("Epoch:", epoch + 1)
            print("Train loss:", logs['loss'])
            print("Train accuracy:", logs['accuracy'])
            print("Test loss:", logs['val_loss'])

    loss_history = LossHistory()

    history = model.fit(data_sequences, labels, validation_split=0.2, epochs=5, batch_size=32, callbacks=[checkpoint, loss_history])

    return history

def predict_test_data_lstm(user):
    db = get_db()
    cursor = db.cursor()

    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])

    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])
    num_words = min(len(word_index) + 1, len(glove_model))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(
        Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                  trainable=False))
    model.add(LSTM(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights('glove_lstm_model.h5')


    db = get_db()
    cursor = db.cursor()

    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())

    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    predictions = model.predict(data_sequences)

    db = get_db()
    cursor = db.cursor()

    for i, p in enumerate(predictions):
        record_id = df.loc[i, 'id']
        cursor.execute("UPDATE test_data_seg SET trisk = ? WHERE id = ?", (float(p), str(record_id)))
        db.commit()

    db.commit()
    close_connection(db)

    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]

    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)
    auc = roc_auc_score(states, predictions)

    return accuracy, precision, recall, f1, auc

def train_bilstm(user):
    db = get_db()
    cursor = db.cursor()

    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])

    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])
    num_words = min(len(word_index) + 1, len(glove_model))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(
        Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                  trainable=False))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('glove_bilstm_model.h5', monitor='val_accuracy', mode='max', save_best_only=True,
                                 verbose=1)

    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("Epoch:", epoch + 1)
            print("Train loss:", logs['loss'])
            print("Train accuracy:", logs['accuracy'])
            print("Test loss:", logs['val_loss'])

    loss_history = LossHistory()

    history = model.fit(data_sequences, labels, validation_split=0.2, epochs=5, batch_size=32, callbacks=[checkpoint, loss_history])

    return history

def predict_test_data_bilstm(user):
    db = get_db()
    cursor = db.cursor()

    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])

    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])
    num_words = min(len(word_index) + 1, len(glove_model))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(
        Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                  trainable=False))
    model.add(Bidirectional(LSTM(128, dropout=0.2, recurrent_dropout=0.2)))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights('glove_bilstm_model.h5')

    db = get_db()
    cursor = db.cursor()

    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())

    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    predictions = model.predict(data_sequences)

    db = get_db()
    cursor = db.cursor()

    for i, p in enumerate(predictions):
        record_id = df.loc[i, 'id']
        cursor.execute("UPDATE test_data_seg SET trisk = ? WHERE id = ?", (float(p), str(record_id)))
        db.commit()

    db.commit()
    close_connection(db)

    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]

    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)
    auc = roc_auc_score(states, predictions)

    return accuracy, precision, recall, f1, auc




# train_bilstm("2")
# print(predict_test_data_bilstm("2"))


def train_textrnn(user):
    db = get_db()
    cursor = db.cursor()

    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])

    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])
    num_words = min(len(word_index) + 1, len(glove_model))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2, return_sequences=True))
    model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('glove_textrnn_model.h5', monitor='val_accuracy', mode='max', save_best_only=True,
                                 verbose=1)

    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("Epoch:", epoch + 1)
            print("Train loss:", logs['loss'])
            print("Train accuracy:", logs['accuracy'])
            print("Test loss:", logs['val_loss'])

    loss_history = LossHistory()

    history = model.fit(data_sequences, labels, validation_split=0.2, epochs=5, batch_size=32, callbacks=[checkpoint, loss_history])

    return history

def predict_test_data_textrnn(user):
    db = get_db()
    cursor = db.cursor()

    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])

    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])
    num_words = min(len(word_index) + 1, len(glove_model))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(
        Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                  trainable=False))
    model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2,
                  return_sequences=True))
    model.add(GRU(128, dropout=0.2, recurrent_dropout=0.2))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights('glove_textrnn_model.h5')

    db = get_db()
    cursor = db.cursor()

    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())

    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    predictions = model.predict(data_sequences)

    db = get_db()
    cursor = db.cursor()

    for i, p in enumerate(predictions):
        record_id = df.loc[i, 'id']
        cursor.execute("UPDATE test_data_seg SET trisk = ? WHERE id = ?", (float(p), str(record_id)))
        db.commit()

    db.commit()
    close_connection(db)

    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]

    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)
    auc = roc_auc_score(states, predictions)

    return accuracy, precision, recall, f1, auc





def train_cnn(user):
    db = get_db()
    cursor = db.cursor()

    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])

    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])
    num_words = min(len(word_index) + 1, len(glove_model))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(Conv1D(128, 2, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 4, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('glove_cnn_model.h5', monitor='val_accuracy', mode='max', save_best_only=True,
                                 verbose=1)
    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("Epoch:", epoch + 1)
            print("Train loss:", logs['loss'])
            print("Train accuracy:", logs['accuracy'])
            print("Test loss:", logs['val_loss'])
    loss_history = LossHistory()

    history = model.fit(data_sequences, labels, validation_split=0.2, epochs=5, batch_size=32,
                        callbacks=[checkpoint, loss_history])

    return history

def predict_test_data_cnn(user):
    db = get_db()
    cursor = db.cursor()

    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])

    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])
    num_words = min(len(word_index) + 1, len(glove_model))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    model = Sequential()
    model.add(
        Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                  trainable=False))
    model.add(Conv1D(128, 2, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 4, activation='relu'))
    model.add(Flatten())
    model.add(Dense(1, activation='sigmoid'))
    model.load_weights('glove_cnn_model.h5')

    db = get_db()
    cursor = db.cursor()

    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())

    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    predictions = model.predict(data_sequences)

    db = get_db()
    cursor = db.cursor()

    for i, p in enumerate(predictions):
        record_id = df.loc[i, 'id']
        cursor.execute("UPDATE test_data_seg SET crisk = ? WHERE id = ?", (float(p), str(record_id)))
        db.commit()

    db.commit()
    close_connection(db)

    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]

    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)
    auc = roc_auc_score(states, predictions)

    return accuracy, precision, recall, f1, auc

def train_text_rcnn(user):
    db = get_db()
    cursor = db.cursor()
    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()
    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))
    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")
    word_index = tokenizer.word_index
    embedding_dim = len(list(glove_model.values())[0])
    num_words = min(len(word_index) + 1, len(glove_model))
    embedding_matrix = np.zeros((num_words, embedding_dim))
    for word, i in word_index.items():
        if i >= num_words:
            continue
        embedding_vector = glove_model.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector

    input_layer = Input(shape=(max_sequence_length,))
    embedding_layer = Embedding(num_words, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                                trainable=False)(input_layer)
    lstm_layer = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
    conv_layer = Conv1D(128, kernel_size=3, padding='valid', kernel_initializer='glorot_uniform')(lstm_layer)
    max_pool_layer = GlobalMaxPooling1D()(conv_layer)
    concatenate_layer = concatenate([max_pool_layer, lstm_layer[:, -1, :]])
    output_layer = Dense(64, activation='relu')(concatenate_layer)
    output_layer = Dropout(0.2)(output_layer)
    output_layer = Dense(1, activation='sigmoid')(output_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('text_rcnn_model.h5', monitor='val_accuracy', mode='max', save_best_only=True,
                                 verbose=1)
    class LossHistory(Callback):
        def on_epoch_end(self, epoch, logs=None):
            print("Epoch:", epoch + 1)
            print("Train loss:", logs['loss'])
            print("Train accuracy:", logs['accuracy'])
            print("Test loss:", logs['val_loss'])

    loss_history = LossHistory()

    history = model.fit(data_sequences, labels, validation_split=0.2, epochs=5, batch_size=32,
                        callbacks=[checkpoint, loss_history])

    return history

def predict_test_data_textrcnn(user):
    db = get_db()
    cursor = db.cursor()
    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()
    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])
    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    model = load_model('text_rcnn_model.h5')

    db = get_db()
    cursor = db.cursor()
    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()

    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())

    sequences = tokenizer.texts_to_sequences(texts)
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    predictions = model.predict(data_sequences)

    for i, p in enumerate(predictions):
        record_id = df.loc[i, 'id']
        cursor.execute("UPDATE test_data_seg SET trisk = ? WHERE id = ?", (float(p), str(record_id)))
        db.commit()

    close_connection(db)

    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]
    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)
    auc = roc_auc_score(states, predictions)

    return accuracy, precision, recall, f1, auc

def train_predict_svm(user):
    db = get_db()
    cursor = db.cursor()
    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()
    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])
    df = df.sample(frac=0.03, random_state=42)

    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    df['comment'] = df['comment'].apply(lambda x: jieba.lcut(str(x)))

    vectors = []
    for comment in df['comment']:
        vec = np.zeros(len(list(glove_model.values())[0]))
        count = 0
        for word in comment:
            if word in glove_model:
                vec += glove_model[word]
                count += 1
        if count != 0:
            vec /= count
        vectors.append(vec)

    X_train = np.array(vectors)
    y_train = np.array(df['state'].values.tolist())

    model = svm.SVC(C=0.001, kernel='linear', probability=True, random_state=42)
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
        vec = np.zeros(len(list(glove_model.values())[0]))
        count = 0
        for word in comment:
            if word in glove_model:
                vec += glove_model[word]
                count += 1
        if count != 0:
            vec /= count
        vectors.append(vec)

    X_test = np.array(vectors)
    y_test = np.array(df['state'].values.tolist())

    predictions = model.predict(X_test)

    db = get_db()
    cursor = db.cursor()
    for i, (index, row) in enumerate(df.iterrows()):
        query = "UPDATE test_data_seg SET trisk = ? WHERE id = ?"
        parameters = (float(predictions[i]), int(row['id']))
        cursor.execute(query, parameters)
    db.commit()
    close_connection(db)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    probabilities = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probabilities)

    print(accuracy, precision, recall, f1, auc)

def train_predict_nb(user):
    db = get_db()
    cursor = db.cursor()
    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()
    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])
    df = df.sample(frac=0.03, random_state=42)

    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")

    df['comment'] = df['comment'].apply(lambda x: jieba.lcut(str(x)))

    vectors = []
    for comment in df['comment']:
        vec = np.zeros(len(list(glove_model.values())[0]))
        count = 0
        for word in comment:
            if word in glove_model:
                vec += glove_model[word]
                count += 1
        if count != 0:
            vec /= count
        vectors.append(vec)

    X_train = np.array(vectors)
    y_train = np.array(df['state'].values.tolist())

    X_train = np.abs(X_train)

    model = MultinomialNB()
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
        vec = np.zeros(len(list(glove_model.values())[0]))
        count = 0
        for word in comment:
            if word in glove_model:
                vec += glove_model[word]
                count += 1
        if count != 0:
            vec /= count
        vectors.append(vec)

    X_test = np.array(vectors)
    y_test = np.array(df['state'].values.tolist())

    X_test = np.abs(X_test)

    predictions = model.predict(X_test)

    db = get_db()
    cursor = db.cursor()
    for i, (index, row) in enumerate(df.iterrows()):
        query = "UPDATE test_data_seg SET trisk = ? WHERE id = ?"
        parameters = (float(predictions[i]), int(row['id']))
        cursor.execute(query, parameters)
    db.commit()
    close_connection(db)

    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)
    probabilities = model.predict_proba(X_test)[:, 1]

    auc = roc_auc_score(y_test, probabilities)

    print(accuracy, precision, recall, f1, auc)


def train_predict_rf(user):

    db = get_db()
    cursor = db.cursor()


    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()


    close_connection(db)


    df = pd.DataFrame(result, columns=['comment', 'state'])
    df = df.sample(frac=0.06, random_state=42)


    glove_model = load_glove_embedding("zhs_wiki_glove.vectors.100d.txt.txt")


    df['comment'] = df['comment'].apply(lambda x: jieba.lcut(str(x)))


    vectors = []
    for comment in df['comment']:
        vec = np.zeros(len(list(glove_model.values())[0]))
        count = 0
        for word in comment:
            if word in glove_model:
                vec += glove_model[word]
                count += 1
        if count != 0:
            vec /= count
        vectors.append(vec)


    X_train = np.array(vectors)
    y_train = np.array(df['state'].values.tolist())


    X_train = np.abs(X_train)


    model = RandomForestClassifier(n_estimators=100, random_state=42)
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
        vec = np.zeros(len(list(glove_model.values())[0]))
        count = 0
        for word in comment:
            if word in glove_model:
                vec += glove_model[word]
                count += 1
        if count != 0:
            vec /= count
        vectors.append(vec)


    X_test = np.array(vectors)
    y_test = np.array(df['state'].values.tolist())


    X_test = np.abs(X_test)


    predictions = model.predict(X_test)


    db = get_db()
    cursor = db.cursor()
    for i, (index, row) in enumerate(df.iterrows()):
        query = "UPDATE test_data_seg SET trisk = ? WHERE id = ?"
        parameters = (float(predictions[i]), int(row['id']))
        cursor.execute(query, parameters)
    db.commit()
    close_connection(db)


    accuracy = accuracy_score(y_test, predictions)
    precision = precision_score(y_test, predictions)
    recall = recall_score(y_test, predictions)
    f1 = f1_score(y_test, predictions)


    probabilities = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, probabilities)

    print(accuracy, precision, recall, f1, auc)

train_predict_rf("2")