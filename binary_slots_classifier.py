import pandas as pd
from collections import defaultdict, Counter
import numpy as np
from sklearn.svm import SVC
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import cross_val_score
from sklearn.manifold.t_sne import TSNE
from nltk.tokenize import sent_tokenize, word_tokenize
from sklearn.externals import joblib
from matplotlib import pyplot as plt
import nlu

slot_names = ['cost_of_service', 'show_docs', 'online_reserving']
pipe = nlu.Pipeline(sent_tokenize, word_tokenize, [nlu.PyMorphyPreproc(), nlu.Lower()], embedder=np.vstack)

def tokenize(text, pipe=pipe):
    _, normed = pipe.feed(text)
    return [w['normal'] for w in normed]


def oversample(X, y):
    """
    :param X: features
    :param y: labels
    :return: new balanced dataset 
            with oversampled minor class 
    
    """
    print('Oversampling...')
    c = Counter(y)
    labels = list(set(y))
    minor_label = min(labels, key=lambda x: c[x])
    print("minor: ", minor_label)
    offset = np.abs(list(c.values())[0] - list(c.values())[1])
    print("offset: ", offset)

    y_new = np.hstack((y, [minor_label] * offset))
    tmp = X[np.array(y) == minor_label]
    sampled = np.random.choice(np.arange(len(tmp)), size=offset)

    X_new = np.vstack((X, tmp[sampled]))
    assert len(X_new) == len(y_new)
    assert np.sum(y_new == minor_label) == len(y_new) // 2

    return X_new, y_new


def extract_features(sents, train=True, save_model=True, filenames=None, tokenizer=tokenize):
    """
    :param sents: iterable with str objects
    :param tokenizer: func feeding row sents and return list of words
    :param train: True if train vectorizers on the sents; else -- filenames of models expected
    :param filenames: dict like {"word":"model path", "char": "model path"} is required if train = False; default: None;
    :param save_model: save vectorizers on the disk if train=True; default: true
    :return: feature matrix
    
    """
    if train:
        words_vectorizer = CountVectorizer(ngram_range=(1, 2), tokenizer=tokenize)  # taking into account pairs of words
        chars_vectorizer = CountVectorizer(analyzer='char_wb',
                                           ngram_range=(2, 4))  # taking into account only n-grams into word boundaries

        if not isinstance(sents, list):
            raise Exception("sents expected to be list of strings")

        mtx_words = words_vectorizer.fit_transform(sents).toarray()  # escape sparse representation
        mtx_chars = chars_vectorizer.fit_transform(sents).toarray()

        if save_model:
            joblib.dump(words_vectorizer, 'words_vectorizer.model')
            joblib.dump(chars_vectorizer, 'chars_vectorizer.model')
        return np.hstack((mtx_words, mtx_chars))


    elif (not train) and isinstance(filenames, dict):
        words_vectorizer = joblib.load(filenames['word'])
        chars_vectorizer = joblib.load(filenames['char'])

        if isinstance(sents, str):
            sents = [sents]

        mtx_words = words_vectorizer.transform(sents).toarray()
        mtx_chars = chars_vectorizer.transform(sents).toarray()

        return np.hstack((mtx_words, mtx_chars))
    else:
        raise Exception("Undefined vectorizers!")


#------------ making train data ---------------#

data = pd.read_csv("./generated_dataset.tsv", sep='\t')
sents = []
y = defaultdict(list)

for i, row in data.iterrows():
    sents.append(row['request'])

    # add targets
    for slot in slot_names:
        y[slot].append(not pd.isnull(row[slot]))

X = extract_features(sents, train=True, save_model=True, filenames=None, tokenizer=tokenize)

#---------------- validate --------------------#

for slot in slot_names:
    print("SLOT: ", slot)
    x_tmp, y_tmp = oversample(X, y[slot])
    svm = SVC()
    scores = cross_val_score(svm, x_tmp, y_tmp, scoring='f1')
    print("For slot: {} cv f1 score: {}".format(slot, scores))
    joblib.dump(svm.fit(x_tmp, y_tmp), '{}_svm_{}.model'.format(slot, np.round(np.mean(scores), 2)))
    print('==Model dumped==')


#---------------- visualize --------------------#


tsne = TSNE()
data_tsne = tsne.fit_transform(X=X)
plt.figure(figsize=(6,5))
for slot in slot_names:
    plt.scatter(data_tsne[y[slot], 0], data_tsne[y[slot], 1], alpha=0.3, label=slot)
plt.legend()
plt.savefig("tsne.png")