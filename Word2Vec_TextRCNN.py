import numpy as np
import pandas as pd
import jieba
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, Bidirectional, LSTM, Conv1D, MaxPooling1D, GlobalMaxPooling1D, Dense, Input
from keras.models import Model
from gensim.models import Word2Vec
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

start_time = time.process_time()

DATABASE = 'weibo_db.db'


def get_db():
    db = sqlite3.connect(DATABASE)
    return db
def close_connection(db):
    db.close()

# Load data and preprocess
def predict_test_data(user):
    # Load and preprocess your data as before
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

    # Tokenization and word embedding
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    max_sequence_length = 180
    embedding_dim = word2vec_model.vector_size
    word_index = tokenizer.word_index
    embedding_matrix = np.zeros((len(word_index) + 1, embedding_dim))
    for word, i in word_index.items():
        if word in word2vec_model.wv:
            embedding_matrix[i] = word2vec_model.wv[word]

    # Create TextRCNN model
    input_layer = Input(shape=(max_sequence_length,), dtype='int32')
    embedding_layer = Embedding(len(word_index) + 1, embedding_dim, weights=[embedding_matrix],
                                input_length=max_sequence_length, trainable=False)(input_layer)
    bi_lstm = Bidirectional(LSTM(128, return_sequences=True))(embedding_layer)
    conv1d = Conv1D(128, 5, activation='relu')(bi_lstm)
    max_pooling = GlobalMaxPooling1D()(conv1d)
    output_layer = Dense(1, activation='sigmoid')(max_pooling)
    model = Model(inputs=input_layer, outputs=output_layer)
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Pad sequences and train the model
    sequences = pad_sequences(sequences, maxlen=max_sequence_length)
    states = df['state'].values
    history = model.fit(sequences, states, epochs=1, batch_size=32, validation_split=0.2)

    # Load test data and preprocess it
    db = get_db()
    cursor = db.cursor()
    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()
    close_connection(db)
    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])

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

end_time = time.process_time()

cpu_time = end_time - start_time
print(f"CPU time: {cpu_time} seconds")
