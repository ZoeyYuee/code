import pandas as pd
import numpy as np
import jieba
from keras.models import Sequential
from keras.layers import Embedding, LSTM, GlobalMaxPooling1D, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from gensim.models import Word2Vec
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
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, GlobalMaxPooling1D, Dense, Dropout, BatchNormalization
from tensorflow.keras.regularizers import l2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import time



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
    states = np.array(df['state'].values.tolist())

    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_sequence_length = 180
    word_index = tokenizer.word_index

    embedding_dim = word2vec_model.vector_size
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))

    for word, i in word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]

    model = Sequential()
    model.add(
        Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix], input_length=max_sequence_length,
                  trainable=False))
    model.add(LSTM(128, return_sequences=True))  # Use LSTM layer instead of Conv1D
    model.add(GlobalMaxPooling1D())
    model.add(Dense(64, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    padded_sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    history = model.fit(padded_sequences,states, epochs=1, batch_size=32, validation_split=0.2)

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
    sequences = tokenizer.texts_to_sequences(texts)
    states = np.array(df['state'].values.tolist())

    # Padding sequences if needed
    sequences = pad_sequences(sequences, maxlen=max_sequence_length)

    # Now, you can use this model for prediction on the test data.
    predictions = model.predict(sequences)
    predicted_states = [1 if p >= 0.5 else 0 for p in predictions]

    accuracy = accuracy_score(states, predicted_states)
    precision = precision_score(states, predicted_states)
    recall = recall_score(states, predicted_states)
    f1 = f1_score(states, predicted_states)

    print(accuracy,precision,recall,f1)

predict_test_data('2')


