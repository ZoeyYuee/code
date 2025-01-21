from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Bidirectional, LSTM
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import sqlite3
import jieba
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from keras.callbacks import ModelCheckpoint

DATABASE = 'weibo_db.db'


def get_db():
    db = sqlite3.connect(DATABASE)
    return db


def close_connection(db):
    db.close()


def predict_test_data(user):
    db = get_db()
    cursor = db.cursor()

    # Querying training data
    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, (user,))
    result = cursor.fetchall()
    close_connection(db)

    train_df = pd.DataFrame(result, columns=['comment', 'state'])
    word2vec_model = Word2Vec.load("word2vec_model.bin")

    train_df['comment'] = train_df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts_train = train_df['comment'].values.tolist()
    labels_train = np.array(train_df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts_train)
    sequences_train = tokenizer.texts_to_sequences(texts_train)

    max_sequence_length = 180
    X_train = pad_sequences(sequences_train, maxlen=max_sequence_length)

    embedding_dim = word2vec_model.vector_size
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(Bidirectional(LSTM(128), merge_mode='concat'))
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    checkpoint = ModelCheckpoint('bilstm_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)

    model.fit(X_train, labels_train, validation_split=0.2, epochs=2, batch_size=32, callbacks=[checkpoint])

    db = get_db()
    cursor = db.cursor()

    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, (user,))
    result = cursor.fetchall()
    close_connection(db)

    test_df = pd.DataFrame(result, columns=['id', 'comment', 'state'])
    test_df['comment'] = test_df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))
    texts_test = test_df['comment'].values.tolist()
    states_test = np.array(test_df['state'].values.tolist())

    sequences_test = tokenizer.texts_to_sequences(texts_test)
    X_test = pad_sequences(sequences_test, maxlen=max_sequence_length)

    predictions = model.predict(X_test)
    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]

    accuracy = accuracy_score(states_test, predicted_states)
    precision = precision_score(states_test, predicted_states)
    recall = recall_score(states_test, predicted_states)
    f1 = f1_score(states_test, predicted_states)

    print("Accuracy:", accuracy)
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1 Score:", f1)

    return accuracy, precision, recall, f1

predict_test_data('2')
