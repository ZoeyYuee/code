import sqlite3
import jieba
import numpy as np
import networkx as nx
import pandas as pd
from decimal import getcontext, Decimal
from gensim.models import Word2Vec
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB

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
    count_vectorizer = CountVectorizer()
    tfidf_transformer = TfidfTransformer()
    X_train_counts = count_vectorizer.fit_transform(texts)
    X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)
    y_train = labels
    model = MultinomialNB()
    model.fit(X_train_tfidf, y_train)
    db = get_db()
    cursor = db.cursor()
    query = "SELECT id, comment, state FROM test_data_seg WHERE user = ? AND is_deleted = 0 AND is_labeled = 1"
    parameters = (user,)
    cursor.execute(query, parameters)
    result = cursor.fetchall()
    close_connection(db)
    df = pd.DataFrame(result, columns=['id', 'comment', 'state'])
    df.dropna(inplace=True)
    df['comment'] = df['comment'].apply(lambda x: ' '.join(jieba.cut(str(x))))
    texts = df['comment'].values.tolist()
    states = np.array(df['state'].values.tolist())
    X_test_counts = count_vectorizer.transform(texts)
    X_test_tfidf = tfidf_transformer.transform(X_test_counts)
    y_test = states
    predictions = model.predict(X_test_tfidf)
    accuracy = accuracy_score(states, predictions)
    precision = precision_score(states, predictions)
    recall = recall_score(states, predictions)
    f1 = f1_score(states, predictions)
    print(accuracy, precision, recall, f1)
    return accuracy, precision, recall, f1

def build_graph_from_relations(user):
    db = get_db()
    cursor = db.cursor()
    cursor.execute("SELECT fan_id, follow_id FROM wb_user_relation WHERE user = ?", (user,))
    relation_data = cursor.fetchall()
    cursor.execute("SELECT SUM(total_wb) AS total_wb_sum FROM wb_userinfo WHERE user = ?", (user,))
    total_wb_sum = cursor.fetchone()[0]
    G = nx.DiGraph()
    for row in relation_data:
        fan_id = int(row[0])
        follow_id = int(row[1])
        cursor.execute("SELECT weibo_id, total_wb, night_wb, followers, following, trisk FROM wb_userinfo WHERE user = ? AND weibo_id = ?", (user, fan_id,))
        fan_data = cursor.fetchone()
        fan_weibo_id = int(fan_data[0])
        fan_total_wb = int(fan_data[1])
        fan_night_wb = int(fan_data[2])
        fan_followers = int(fan_data[3])
        fan_following = int(fan_data[4])
        fan_trisk = float(fan_data[5])
        confidence1 = 1/(fan_followers+fan_following)
        night1 = fan_night_wb/50
        activity1 = fan_total_wb/total_wb_sum
        cursor.execute("SELECT weibo_id, total_wb, night_wb, followers, following, trisk FROM wb_userinfo WHERE user = ? AND weibo_id = ?", (user, follow_id,))
        follow_data = cursor.fetchone()
        follow_weibo_id = int(follow_data[0])
        follow_total_wb = int(follow_data[1])
        follow_night_wb = int(follow_data[2])
        follow_followers = int(follow_data[3])
        follow_following = int(follow_data[4])
        follow_trisk = float(follow_data[5])
        confidence = 1 / (follow_followers + follow_following)
        night = follow_night_wb / 50
        activity = follow_total_wb / total_wb_sum
        if not G.has_node(fan_weibo_id):
            G.add_node(fan_weibo_id, activity=activity1, night=night1, confidence=confidence1, trisk=fan_trisk, prisk=0.0)
        if not G.has_node(follow_weibo_id):
            G.add_node(follow_weibo_id, activity=activity, night=night, confidence=confidence, trisk=follow_trisk, prisk=0.0)
        G.add_edge(fan_weibo_id, follow_weibo_id)
    close_connection(db)
    nx.write_gexf(G, 'graph1.gexf')
    return G

def pagerank_with_suicide(G, alpha=0.85, personalization=None, max_iter=100, tol=1e-6, nstart=None, weight='weight', dangling=None, a = 0.55):
    if len(G) == 0:
        return {}
    if not G.is_directed():
        D = G.to_directed()
    else:
        D = G
    W = nx.stochastic_graph(D, weight=weight)
    N = W.number_of_nodes()
    if nstart is None:
        x = dict.fromkeys(W, 1.0 / N)
    else:
        s = float(sum(nstart.values()))
        x = {k: v / s for k, v in nstart.items()}
    if personalization is None:
        p = dict.fromkeys(W, 1.0 / N)
    else:
        missing = set(G) - set(personalization)
        if missing:
            raise nx.NetworkXError(f"Personalization dictionary must have a value for every node. Missing nodes: {missing}")
        s = float(sum(personalization.values()))
        p = {k: v / s for k, v in personalization.items()}
    if dangling is None:
        dangling_weights = p
    else:
        missing = set(G) - set(dangling)
        if missing:
            raise nx.NetworkXError(f"Dangling node dictionary must have a value for every node. Missing nodes: {missing}")
        s = float(sum(dangling.values()))
        dangling_weights = {k: v / s for k, v in dangling.items()}
    dangling_nodes = [n for n in W if W.out_degree(n, weight=weight) == 0.0]
    for _ in range(max_iter):
        xlast = x
        x = dict.fromkeys(xlast.keys(), 0)
        danglesum = alpha * sum(xlast[n] for n in dangling_nodes)
        for n in x:
            for nbr in W[n]:
                x[nbr] += alpha * xlast[n] * W[n][nbr][weight] * G.nodes[nbr]['trisk']
            x[n] += danglesum * dangling_weights[n] + (1.0 - alpha) * p[n] * G.nodes[n]['trisk']
        err = sum([abs(x[n] - xlast[n]) for n in x])
        if err < N * tol:
            return x
        nx.write_gexf(G, 'g.gexf')
    raise nx.NetworkXError(f"pagerank: power iteration failed to converge in {max_iter} iterations.")

def calculate_node_suicide_risk(G, b, pagerank_dict):
    getcontext().prec = 200
    for node in G.nodes:
        suicide_risk = Decimal(G.nodes[node]['trisk'])
        pr_sum = Decimal(0)
        pr_sum_neighbors = Decimal(0)
        for neighbor in G.neighbors(node):
            neighbor_suicide = Decimal(G.nodes[neighbor]['trisk'])
            pr_value = Decimal(pagerank_dict[neighbor])
            pr_sum_neighbors += pr_value
            pr_sum += neighbor_suicide * pr_value
        if pr_sum_neighbors != Decimal(0):
            avgrisk = pr_sum / pr_sum_neighbors
        else:
            avgrisk = Decimal(0)
        if avgrisk == 0:
            node_risk = suicide_risk
        else:
            node_risk = Decimal(b) * Decimal(suicide_risk) + Decimal(1 - b) * Decimal(avgrisk)
        G.nodes[node]['prisk'] = node_risk

def save_prisk_to_db(G):
    db = get_db()
    cursor = db.cursor()
    for node in G.nodes():
        prisk = G.nodes[node]['prisk']
        prisk_rounded = round(prisk, 4)
        prisk_str = str(prisk_rounded)
        cursor.execute("UPDATE wb_userinfo SET prisk = ? WHERE weibo_id = ?", (prisk_str , int(node)))
    db.commit()
    close_connection(db)

def main(user):
    accuracy, precision, recall, f1 = predict_test_data(user)
    G = build_graph_from_relations(user)
    pagerank_dict = pagerank_with_suicide(G, alpha=0.85)
    b = 0.55
    calculate_node_suicide_risk(G, b, pagerank_dict)
    save_prisk_to_db(G)
    return accuracy, precision, recall, f1
