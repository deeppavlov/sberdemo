from collections import Counter
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.base import TransformerMixin
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
    print()

    X_new = np.concatenate((X_new, np.array(sampled_data)))
    assert len(X_new) == len(c.keys()) * c[major_label]
    assert len(X_new) == len(y_new)

    return X_new, y_new


class FeatureExtractor(TransformerMixin):
    def __init__(self, use_chars=False):
        self._been_fitten = False
        self.use_chars = use_chars
        self.words_vectorizer = TfidfVectorizer(ngram_range=(1, 2), )  # taking into account pairs of words
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
