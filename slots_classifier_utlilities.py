from collections import defaultdict, Counter
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.base import TransformerMixin
import nlu


def normalize_tokenizer(text):
    pipe = nlu.Pipeline_nlp(sent_tokenize, word_tokenize, [nlu.PyMorphyPreproc(), nlu.Lower()], embedder=np.vstack)
    _, normed = pipe.feed(text)
    return [w['normal'] for w in normed]


def oversample(X, y, verbose=False):
    """
    :param X: features
    :param y: labels
    :return: new balanced dataset 
            with oversampled minor class 

    """
    if verbose:
        print('Oversampling...')
    c = Counter(y)
    labels = list(set(y))
    minor_label = min(labels, key=lambda x: c[x])
    if verbose:
        print("minor: ", minor_label)
    offset = np.abs(list(c.values())[0] - list(c.values())[1])
    if verbose:
        print("offset: ", offset)

    y_new = np.hstack((y, [minor_label] * offset))
    tmp = X[np.array(y) == minor_label]
    sampled = np.random.choice(np.arange(len(tmp)), size=offset)

    X_new = np.vstack((X, tmp[sampled]))
    assert len(X_new) == len(y_new)
    assert np.sum(y_new == minor_label) == len(y_new) // 2

    return X_new, y_new


class FeatureExtractor(TransformerMixin):
    def __init__(self, use_chars=False, tokenizer=normalize_tokenizer):
        self.tokenizer = tokenizer
        self._been_fitten = False
        self.use_chars = use_chars
        self.words_vectorizer = CountVectorizer(ngram_range=(1, 2),
                                                tokenizer=self.tokenizer)  # taking into account pairs of words
        if self.use_chars:
            self.chars_vectorizer = CountVectorizer(analyzer='char_wb',
                                                    ngram_range=(2, 4))  # taking into account
            # only n-grams into word boundaries
        else:
            self.chars_vectorizer = None

    def fit(self, raw_docs, y = None):
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
