import os
from sklearn.feature_extraction.text import CountVectorizer as Vectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.tree import DecisionTreeClassifier as Model
from scipy.sparse import hstack
import shutil
import time
import pickle

from lgid.features import (
    get_instances
)

t1 = time.time()


def train(infiles, modelpath, config, monolingual=None):
    global t1
    if os.path.exists(modelpath):
        shutil.rmtree(modelpath)
    os.makedirs(modelpath)
    texts, labels, others, ids = get_instances(infiles, config)
    print('Getting instances: ' + str(time.time() - t1))
    t1 = time.time()
    char_max = int(config['parameters']['character-n-gram-size'])
    char_count = Vectorizer(texts, ngram_range=(1, char_max), analyzer='char').fit(texts)
    word_max = int(config['parameters']['word-n-gram-size'])
    word_count = Vectorizer(texts, ngram_range=(1, word_max), analyzer='word').fit(texts)
    feat_vectr = DictVectorizer().fit(others)
    print('Training vectorizers: ' + str(time.time() - t1))
    t1 = time.time()
    char_matrix = char_count.transform(texts)
    word_matrix = word_count.transform(texts)
    other_matrix = feat_vectr.transform(others)
    pickle.dump(char_count, open(modelpath + '/char.p', 'wb'))
    pickle.dump(word_count, open(modelpath + '/word.p', 'wb'))
    pickle.dump(feat_vectr, open(modelpath + '/feats.p', 'wb'))
    main_x = hstack([char_matrix, word_matrix, other_matrix])
    labels = labels
    model = Model()
    model.fit(main_x, labels)
    print('Fitting model: ' + str(time.time() - t1))
    pickle.dump(model, open(modelpath + '/model.p', 'wb'))
    t1 = time.time()


def classify(texts, others, modelpath):
    """
    Classify a list of strings by language

    :param texts: list of language strings
    :param others: list of feature dicts corresponding to the strings
    :param modelpath: filepath of model folder
    :return: list of language labels (string)
    """
    model = pickle.load(open(modelpath + '/model.p', 'rb'))
    char_counts = pickle.load(open(modelpath + '/char.p', 'rb'))
    word_counts = pickle.load(open(modelpath + '/word.p', 'rb'))
    feat_vectr = pickle.load(open(modelpath + '/feats.p', 'rb'))

    char_matrix = char_counts.transform(texts)
    word_matrix = word_counts.transform(texts)
    other_matrix = feat_vectr.transform(others)

    main_x = hstack([char_matrix, word_matrix, other_matrix])
    result = model.predict(main_x)
    return result