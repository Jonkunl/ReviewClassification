from gensim.models.keyedvectors import KeyedVectors
import numpy as np
import ast
from sklearn.feature_extraction.text import TfidfVectorizer

def w2v_embedding(model_path, reviews):
    w2v_model = KeyedVectors.load_word2vec_format(model_path, binary=True)
    output = []
    for i in reviews:
        for word in i.split():
            i_vec = []
            if word in w2v_model.vocab:
                i_vec.append(w2v_model[word])
            else:
                i_vec.append(np.zeros(300))

        output.append(np.mean(i_vec, axis=0))

    return output

def s2v_str2vec(string):
    str1 = string.replace('\n', '').replace('[ ', '[').replace(' ]', ']')
    str2 = ','.join(str1.split())
    return ast.literal_eval(str2)


def s2v_embedding(review_vectors):
    s2v_embed = []
    for sv in review_vectors:
        s2v_embed.append(s2v_str2vec(sv))

    return np.asarray(s2v_embed)


def bert_embedding(review_vectors):
    bert_embed = []
    for sv in review_vectors:
        bert_embed.append(ast.literal_eval(sv))

    return np.asarray(bert_embed)


def tfidf_embedding(reviews):
    tfidf = TfidfVectorizer(max_features=300)
    output = tfidf.fit_transform(reviews)
    return output
