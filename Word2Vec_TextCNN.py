from gensim.models import Word2Vec
from keras.models import Sequential
from keras.layers import Dense, Dropout, Embedding, Conv1D, GlobalMaxPooling1D
from keras.preprocessing.text import Tokenizer
from keras.src.utils import pad_sequences
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

    query = "SELECT comment, state FROM train_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    cursor.execute(query, user)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['comment', 'state'])

    word2vec_model = Word2Vec.load("word2vec_model.bin")

    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    labels = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    max_sequence_length = 180

    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    embedding_dim = word2vec_model.vector_size
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]

    model = Sequential()
    model.add(Embedding(len(word_index) + 1, 100, weights=[embedding_matrix], input_length=max_sequence_length,
                        trainable=False))
    model.add(Conv1D(128, 2, activation='relu'))
    model.add(Conv1D(128, 3, activation='relu'))
    model.add(Conv1D(128, 4, activation='relu'))
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    # model.load_weights('text_cnn_model.h5')

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    checkpoint = ModelCheckpoint('text_cnn_model.h5', monitor='val_accuracy', mode='max', save_best_only=True, verbose=1)
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    history = model.fit(padded_sequences, labels, validation_split=0.2, epochs=5, batch_size=32, callbacks=[checkpoint])

    db = get_db()
    cursor = db.cursor()

    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()

    close_connection(db)

    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])

    # word2vec_model = Word2Vec.load("word2vec_model.bin")
    #
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))

    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())

    # tokenizer = Tokenizer()
    # tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    #
    max_sequence_length = 180
    #
    data_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    #
    # model = load_model('text_cnn_model.h5')
    #
    predictions = model.predict(data_sequences)
    risk = [p[0] for p in predictions]
    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]

    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)

    print(accuracy,precision,recall,f1)

    return accuracy, precision, recall, f1

predict_test_data('2')
