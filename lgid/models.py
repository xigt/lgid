
# This module is adapted from the one in this repository:
#   https://github.com/xigt/classifier-common
# There is no associated license, but it is part of the same Xigt
# organization, whose projects generally use the MIT license.

import gzip
import logging
import os
import pickle
# import multiprocessing

from sklearn.ensemble import AdaBoostClassifier
from sklearn.feature_extraction.text import CountVectorizer as Vectorizer
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.tree import DecisionTreeClassifier as Model
from scipy.sparse import hstack
import shutil
import time
import pickle

import numpy as np
from sklearn.linear_model import LogisticRegression
# from abc import abstractclassmethod

LOG = logging.getLogger()

def LM_train(texts, labels, modelpath, config):
    if os.path.exists(modelpath):
        shutil.rmtree(modelpath)
    os.makedirs(modelpath)
    char_max = int(config['parameters']['character-n-gram-size'])
    char_count = Vectorizer(texts, ngram_range=(1, char_max), analyzer='char').fit(texts)
    word_max = int(config['parameters']['word-n-gram-size'])
    word_count = Vectorizer(texts, ngram_range=(1, word_max), analyzer='word').fit(texts)
    char_matrix = char_count.transform(texts)
    word_matrix = word_count.transform(texts)
    pickle.dump(char_count, open(modelpath + '/char.p', 'wb'))
    pickle.dump(word_count, open(modelpath + '/word.p', 'wb'))
    main_x = hstack([char_matrix, word_matrix])
    labels = labels
    model = Model()
    model.fit(main_x, labels)
    pickle.dump(model, open(modelpath + '/model.p', 'wb'))



class DataInstance(object):
    def __init__(self, label, feats):
        self.label = label
        self.feats = feats

class DocInstance(DataInstance):
    """
    Wrapper class to hold the path, doc_id, label, and features
    """

    def __init__(self, doc_id, label, feats, path):
        super().__init__(label, feats)
        self.path = path
        self.doc_id = doc_id

class StringInstance(DataInstance):
    def __init__(self, id, label, feats):
        super().__init__(label, feats)
        self.id = id

class Distribution(object):
    def __init__(self, classes, probs):
        self.dict = {}
        self.best_class = None
        self.best_prob = 0.0
        for c, p in zip(classes, probs):
            self.dict[c] = p
            if p > self.best_prob:
                self.best_class = c
                self.best_prob = p

    def get(self, key, default=None):
        return self.dict.get(key, None)

    def classes(self): return self.dict.keys()



class ClassifierWrapper(object):
    """
    This class implements a wrapper class to combine sklearn's
    learner, vectorizer, and feature selector classes into one
    serializable object.
    """
    def __init__(self):
        self.dv = DictVectorizer(dtype=int)
        self.feat_selector = None
        self.learner = None

    def _vectorize(self, data, testing=False):
        if testing:
            return self.dv.transform(data)
        else:
            return self.dv.fit_transform(data)

    def _vectorize_and_select(self, data, labels, num_feats=None, testing=False):

        # Start by vectorizing the data.
        vec = self._vectorize(data, testing=testing)

        # Next, filter the data if in testing mode, according
        # to whatever feature selector was defined during
        # training.
        if testing:

            #if self.feat_selector is not None:
                # LOG.info('Feature selection was enabled during training, limiting to {} features.'.format(
                #     self.feat_selector.k))
                #return self.feat_selector.transform(vec)
                
            #else:
            return vec

        # Only do feature selection if num_feats is positive, and there are more features
        # than max_num
        elif num_feats is not None and (num_feats > 0) and num_feats < vec.shape[1]:
            LOG.info('Feature selection enabled, limiting to {} features.'.format(num_feats))
            self.feat_selector = SelectKBest(chi2, num_feats)
            return self.feat_selector.fit_transform(vec, labels)

        else:
            LOG.info("Feature selection disabled, all available features are used.")
            return vec

    def _checklearner(self):
        if self.learner is None:
            raise Exception("Learner must be specified.")

    def train(self, data, num_feats=None, weight_path=None):
        """
        :type data: list[DataInstance]
        """
        self._checklearner()
        labels = [d.label for d in data]
        feats = [d.feats for d in data]

        vec = self._vectorize_and_select(feats, labels, num_feats=num_feats, testing=False)
        self.learner.fit(vec, labels)
        if weight_path is not None:
            LOG.info('Writing feature weights to "{}"'.format(weight_path))
            self.dump_weights(weight_path)

    def test(self, data):
        """
        Given a list of document instances, return a list
        of the probabilities of the Positive, Negative examples.

        :type data: Iterable[DataInstance]
        :rtype: list[Distribution]

        """
        self._checklearner()
        labels = []
        feats = []
        # We need to make this loop happen this way in case
        # the data is a generator, and doing list
        # comprehensions will result in one list being empty.
        for datum in data:
            # vec = self._vectorize_and_select([datum.feats], [datum.label], testing=True)
            # probs = self.learner.predict_proba(vec)
            # yield Distribution(self.classes(), probs[0])
            labels.append(datum.label)
            feats.append(datum.feats)

        vec = self._vectorize_and_select(feats, labels, testing=True)
        #
        # Return the
        probs = self.learner.predict_proba(vec)
        #
        return [Distribution(self.classes(), p) for p in probs]

    def classes(self):
        self._checklearner()
        return self.learner.classes_.tolist()

    def feat_names(self):
        return np.array(self.dv.get_feature_names())

    def feat_supports(self):
        if self.feat_selector is not None:
            return self.feat_selector.get_support()
        else:
            return np.ones((len(self.dv.get_feature_names())), dtype=bool)

    def weights(self):
        """
        Get a list of features and their importances,
        either for logistic regression or adaboost.

        :return:
        """
        if isinstance(self.learner, AdaBoostClassifier):
            feat_weights = self.learner.feature_importances_
            return {f: feat_weights[j] for j, f in enumerate(self.feat_names()[self.feat_supports()])
                    if feat_weights[j] != 0}
        elif isinstance(self.learner, LogisticRegression):
            return {f: self.learner.coef_[0][j] for j, f in enumerate(self.feat_names()[self.feat_supports()])}

    def dump_weights(self, path, n=-1):
        with open(path, 'w') as f:
            sorted_weights = sorted(self.weights().items(), reverse=True, key=lambda x: x[1])
            for feat_name, weight in sorted_weights[:n]:
                f.write('{}\t{}\n'.format(feat_name, weight))

    def save(self, path):
        """
        Serialize the classifier out to a file.
        """
        if os.path.dirname(path): os.makedirs(os.path.dirname(path), exist_ok=True)
        f = gzip.GzipFile(path, 'w')
        pickle.dump(self, f)
        f.close()

    @classmethod
    def load(cls, path):
        f = gzip.GzipFile(path, 'r')
        c = pickle.load(f)
        assert isinstance(c, ClassifierWrapper)
        return c


class LogisticRegressionWrapper(ClassifierWrapper):
    def __init__(self):
        super().__init__()
        self.learner = LogisticRegression()

class AdaboostWrapper(ClassifierWrapper):
    def __init__(self):
        super().__init__()
        self.learner = AdaBoostClassifier()

