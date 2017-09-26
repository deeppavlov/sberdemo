from collections import Counter
import numpy as np
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
from sklearn.linear_model.base import LinearClassifierMixin
from sklearn.svm import LinearSVC
from sklearn.base import BaseEstimator
from sklearn.externals import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from typing import Dict, List, Any, Union
from copy import deepcopy
from numpy.random import RandomState


def oversample_data(X, y, verbose=False, seed=23):

    """
    :param X: features
    :param y: labels
    :return: new balanced dataset 
            with oversampled minor class 
    """

    random_state = RandomState(seed=seed)
    y_new = deepcopy(y)
    X_new = deepcopy(X)

    if verbose:
        print('Oversampling...')
    c = Counter(y)
    labels = list(set(y))
    major_label = max(labels, key=lambda x: c[x])
    if verbose:
        print("major: {}".format(major_label))

    assert major_label == c.most_common()[0][0]

    sampled_data = []
    for label, count in c.most_common()[1:]:
        offset = c[major_label] - count
        y_new = np.hstack((y_new, [label] * offset))
        tmp = X[np.array(y) == label]
        sampled = random_state.choice(np.arange(len(tmp)), size=offset)
        # if isinstance(X[0][0], dict):
        sampled_data.extend(tmp[sampled])
        # else:
        #     sampled_data.extend()X_new = np.vstack((X_new, tmp[sampled]))

        if verbose:
            print("offset: {} for label: {}".format(offset, label))

    X_new = np.concatenate((X_new, np.array(sampled_data)))
    assert len(X_new) == len(c.keys()) * c[major_label]
    assert len(X_new) == len(y_new)

    return X_new, y_new


class FeatureExtractor(TransformerMixin):
    def __init__(self, use_chars=False, stop_words=None):
        self._been_fitten = False
        self.stop_words = stop_words
        self.use_chars = use_chars
        # taking into account pairs of words
        self.words_vectorizer = TfidfVectorizer(ngram_range=(1, 2), stop_words=self.stop_words)
        # TODO: it breaks, why?
        # self.words_vectorizer = TfidfVectorizer(ngram_range=(1, 2, 3), stop_words=self.stop_words)
        if self.use_chars:
            self.chars_vectorizer = TfidfVectorizer(analyzer='char_wb',
                                                    ngram_range=(2, 4))  # taking into account
            # only n-grams into word boundaries
        else:
            self.chars_vectorizer = None

    def fit(self, raw_docs, y=None):
        """
        :param raw_docs: iterable with strings 
        :return: None
        
        """
        self.words_vectorizer.fit(raw_docs)
        if self.use_chars:
            self.chars_vectorizer.fit(raw_docs)

    def fit_transform(self, raw_docs, y=None):
        """
        :param raw_docs: iterable with strings 
        :return: matrix of features
         
        """
        self._been_fitten = True

        if (not isinstance(raw_docs, list)) or (not isinstance(raw_docs[0], str)):
            raise Exception("raw_docs expected to be list of strings")

        mtx_words = self.words_vectorizer.fit_transform(raw_docs).toarray()  # escape sparse representation
        if self.use_chars:
            mtx_chars = self.chars_vectorizer.fit_transform(raw_docs).toarray()
            return np.hstack((mtx_words, mtx_chars))
        else:
            return mtx_words

    def transform(self, raw_docs):
        """
        :param raw_docs: str or iterable with str elements
        :return: matrix with shape: [num_elements, num_features] 
        
        """

        if not self._been_fitten:
            raise Exception("It is necessary to fit before transform")

        # case if one sample
        if isinstance(raw_docs, str):
            raw_docs = [raw_docs]

        mtx_words = self.words_vectorizer.transform(raw_docs).toarray()  # escape sparse representation
        if self.use_chars:
            mtx_chars = self.chars_vectorizer.transform(raw_docs).toarray()
            return np.hstack((mtx_words, mtx_chars))
        else:
            return mtx_words


class StickSentence(TransformerMixin):
    @staticmethod
    def _preproc(data):
        if not isinstance(data[0], list):
            data = [data]
        if isinstance(data[0][0], dict):
            return [" ".join([w['normal'] for w in sent]) for sent in data]
        else:
            return [" ".join(sent) for sent in data]

    def fit_transform(self, data, y=None):
        return self._preproc(data)

    def transform(self, data, y=None):
        return self._preproc(data)


class Embedder(TransformerMixin):
    def __init__(self, fasttext, stop_words=()):
        self.fasttext = fasttext
        self.words_vectorizer = TfidfVectorizer(ngram_range=(1, 1), stop_words=stop_words)
        self.default_k = 1.0

    def _normalize(self, unnormalized: List[Dict]):
        return [w['normal'] for w in unnormalized]

    def fit(self, data: Union[List[List[Dict]], List[Dict]], y=None):
        if not isinstance(data[0], list):
            data = [data]
        self.words_vectorizer.fit([' '.join(self._normalize(d)) for d in data], y)
        return self

    def transform(self, data, y=None):
        if not isinstance(data[0], list):
            data = [data]
        res = []
        for row in data:
            vecs = []
            word2id = self.words_vectorizer.vocabulary_
            id2idf = self.words_vectorizer.idf_
            for x, nw in zip(row, self._normalize(row)):
                v = self.fasttext[x['_text']]
                wid = word2id.get(nw, None)
                if wid is None:
                    k = self.default_k
                else:
                    k = id2idf[wid]
                vecs.append(k * v)
            res.append(np.array(vecs).sum(axis=0))
        return np.vstack(res)


class SentenceClassifier:
    def __init__(self, base_clf=None,
                 stop_words=None, use_chars=False, labels_list=None, model_path=None):
        """
        :param stop_words: list of words to exclude from feature matrix
        :param use_chars: default False
        :param labels_list: list of possible targets; optional
        :param model_path: path to load model from

        """
        # TODO: make this ugly thing more acceptable
        self._initialization(base_clf=base_clf, stop_words=stop_words,
                             use_chars=use_chars, labels_list=labels_list,
                             model_path=model_path)

    def _initialization(self, base_clf=None,
                        stop_words=None, use_chars=False, labels_list=None, model_path=None):

        if base_clf is None:
            base_clf=LinearSVC(C=1)
        self.model = None
        self.use_chars = use_chars
        self.stop_words = stop_words

        if labels_list is not None:
            self.labels_list = sorted(list(set(labels_list)))
            self.string2idx = {s: i for i, s in enumerate(self.labels_list)}
            self.idx2string = {v: k for k, v in self.string2idx.items()}
        else:
            self.labels_list = None

        self.feat_generator = FeatureExtractor(use_chars=use_chars, stop_words=self.stop_words)

        if isinstance(base_clf, BaseEstimator):
            self.clf = base_clf
        else:
            raise Exception("Wrong classifier type")

        self.model = Pipeline([('sticker_sent', StickSentence()),
                               ('feature_extractor', self.feat_generator),
                               ('classifier', self.clf)])
        if model_path is not None:
            self.load_model(model_path)

    def train_model(self, X: List[List[Dict[str, Any]]], y: List,
                    base_clf=LogisticRegression(penalty='l1', C=10),
                    stop_words=None, use_chars=False, labels_list=None):

        self._initialization(base_clf=base_clf, stop_words=stop_words,
                             use_chars=use_chars, labels_list=labels_list)

        if None in y:
            y = [i if i is not None else '_' for i in y]

        if self.labels_list is not None:
            y_idx = [self.string2idx[label] for label in y]
        else:
            self.labels_list = sorted(list(set(y)))
            self.string2idx = {s: i for i, s in enumerate(self.labels_list)}
            self.idx2string = {v: k for k, v in self.string2idx.items()}

            if '_' in self.string2idx:
                self.idx2string[self.string2idx['_']] = None
                self.string2idx[None] = self.string2idx['_']

            y_idx = [self.string2idx[label] for label in y]

        self.model.fit(X, y_idx)

    def predict_single(self, text: List[Dict[str, Any]]):
        assert self.model, 'No model specified!'
        label = self.model.predict(text)[0]
        return self.idx2string[label]

    def predict_batch(self, list_texts: List[List[Dict[str, Any]]]):
        assert self.model, 'No model specified!'
        labels = self.model.predict(list_texts)
        return labels

    def dump_model(self, model_path):
        dump_dict = {'model': self.model,
                     'string2idx': self.string2idx,
                     'stop_words': self.stop_words,
                     'use_chars': self.use_chars}
        joblib.dump(dump_dict, model_path)

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise Exception("Model path: '{}' doesnt exist".format(model_path))
        dump_dict = joblib.load(model_path)
        self.model = dump_dict['model']
        self.clf = self.model.steps[2][1]
        self.string2idx = dump_dict['string2idx']
        self.stop_words = dump_dict['stop_words']
        self.use_chars = dump_dict['use_chars']
        self.labels_list = sorted(list(set(self.string2idx.keys())-{None}))
        self.idx2string = {v: k for k, v in self.string2idx.items()}
        if '_' in self.labels_list:
            self.idx2string[self.string2idx['_']] = None
            self.string2idx[None] = self.string2idx['_']

    def get_feature_importance(self):
        if isinstance(self.clf, LinearClassifierMixin):
            coefs = self.clf.coef_
            names = self.model.steps[1][1].words_vectorizer.get_feature_names()
            results = []
            for line in coefs:
                weights = sorted(list(zip(names, line)), key=lambda x: x[1], reverse=True)
                results.append(weights)
            return results

    def get_description(self):
        descr = str(type(self.clf))
        params = sorted(['{}: {}'.format(repr(k), repr(v)) for k, v in self.clf.get_params().items()])
        params = '{{{}}}'.format(', '.join(params))
        result = "{}\n{}\nstop_words: {}\nuse_chars: {}".format(descr, params, self.stop_words, self.use_chars)
        return result

    def get_labels(self):
        return self.labels_list

    def encode2idx(self, labels):
        return [self.string2idx[w] for w in labels]

    def encode2string(self, indexes):
        return [self.idx2string[i] for i in indexes]