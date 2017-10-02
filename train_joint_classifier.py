import pickle
from itertools import chain
from operator import itemgetter

import pandas as pd
import sys
from scipy.optimize import fmin
from scipy.sparse import spmatrix
from sklearn.linear_model import ElasticNet
from sklearn.metrics import f1_score, precision_recall_fscore_support

from nlu import *
from sklearn.model_selection import GroupKFold
from slots import read_slots_from_tsv, ClassifierSlot

import os
import argparse

import torch
import torch.sparse
import torch.tensor as T
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

THRESHOLD = 0.3

MODEL_FILE = 'nn.model'

DUMP_DEFAULT = True
MODEL_FOLDER_DEFAULT = './models_nlu'
USE_CHAR_DEFAULT = False
COMMON_STOP_WORDS = ['здравствовать', 'добрый', 'день', 'для', 'хотеть', 'нужный', 'бы', 'ли', 'не', 'через', 'без', 'это', 'при', 'по', 'на', 'вечер']
STOP_WORDS_INTENT = []
STOP_WORDS_SLOTS = {'online_reserving':['открыть', 'счет', 'возможно', 'приходить', 'банк', 'как'],

                    'show_docs':['открытие', 'счет', 'какой', 'нужный', 'необходимый', 'сбербанк', 'хотеть', 'открыть'],

                    'cost_of_service':['рассказать', 'открытие', 'хотеть', 'открыть', 'сказать', 'какой', 'счет', 'мочь', 'счет', 'сбер'],

                    'show_phone':['график', 'пожалуйста', 'можно', 'работать',
                                  'офис', 'строгино', 'банк', 'работать', 'сказать', 'ленинский',
                                  'отделение', 'банка' 'ряд', 'чертановский', 'где', 'ближний', 'банк', 'день'],

                    'show_schedule':['телефон', 'работа', 'ряд', 'офис'],

                    'search_vsp':['открыть', 'счет'],

                    'not_first':['открыть', 'что', 'заявление', 'мочь', 'необходимый', 'комплект', 'документ', 'счет']}


need_cross_validation = False


def batch_generator(n, batch_size):
    indexi = np.arange(n)
    while True:
        np.random.shuffle(indexi)
        for i in range(0, n, batch_size):
            yield Variable(torch.LongTensor(indexi[i:i+batch_size]))


def joint_oversampling_coefs(targets, verbose=False):
    X = Variable(torch.FloatTensor(targets))
    w = nn.Parameter(torch.ones(X.size()[0], 1))
    optimizer = optim.SGD([w], lr=0.1)

    mean = Variable(X.sum(0).max().data)

    losses = []
    for epoch in range(1000):
        optimizer.zero_grad()
        counts = X.t() @ w

        loss_not_mean = (counts - mean).abs().mean()
        small_w = w[(w < 1).data]

        if small_w.dim() > 0:
            loss_less_than_one = (1-small_w).sum()
        else:
            loss_less_than_one = Variable(torch.zeros(1))

        loss = loss_not_mean + loss_less_than_one

        losses.append(loss.data[0])

        if verbose:
            print(epoch, '{:.4f}'.format(loss_not_mean.data[0]), '{:.4f}'.format(loss_less_than_one.data[0]), sep='\t')

        loss.backward()
        optimizer.step()

    balancing_counts = w.data.numpy().round().astype(int)
    balancing_idx = np.concatenate([np.repeat(i, b[0]) for i, b in enumerate(balancing_counts)])
    return balancing_idx


class MLPJointclassifier(BaseEstimator):
    def __init__(self, hidden_layer_neurons=40, l1=0.0, l2=0.0, tol=1e-3, batch_size=1, labels=None):
        super().__init__()
        self.hidden_layer_neurons = hidden_layer_neurons
        self.l1 = l1
        self.l2 = l2
        self.tol = tol
        self.batch_size = batch_size
        self.labels = labels

    def predict(self, X):
        return self.predict_proba(X) > 0.5

    def predict_proba(self, X):
        X = X.toarray() if isinstance(X, spmatrix) else X
        return self.model.forward(Variable(torch.FloatTensor(X))).data.numpy()

    def fit(self, X: Union[np.ndarray, spmatrix], Y: np.ndarray, class_weights: np.ndarray=None):
        if self.labels is None:
            self.labels = [str(i) for i in range(Y.shape[1])]
        assert len(self.labels) == Y.shape[1]

        if class_weights is None:
            class_weights = np.ones(Y.shape[1])

        assert X.ndim == 2
        assert Y.ndim == 2
        W1 = nn.Linear(X.shape[1], self.hidden_layer_neurons)
        W2 = nn.Linear(self.hidden_layer_neurons, Y.shape[1])

        X = Variable(torch.FloatTensor(X.toarray() if isinstance(X, spmatrix) else X))
        Y = Variable(torch.FloatTensor(Y))
        W = Variable(torch.FloatTensor(class_weights[np.newaxis]))

        self.model = nn.Sequential(W1, nn.ReLU(), W2, nn.Sigmoid())

        optimizer = optim.Adam(self.model.parameters(), weight_decay=self.l2)

        losses = []
        samples_seen = 0
        for idx in batch_generator(X.size()[0], self.batch_size):
            batch_x = X[idx]
            batch_y = Y[idx]

            predicted_y = self.model.forward(batch_x)
            optimizer.zero_grad()
            batch_loss = batch_y*torch.log(predicted_y)*W + (1-batch_y)*torch.log(1-predicted_y)*W
            loss = -batch_loss.mean(0).sum()
            l1_loss = W1.weight.abs().mean()

            losses.append(loss.data[0])
            samples_seen += batch_x.size()[0]
            print(samples_seen, loss.data[0], np.mean(losses[-10:]), sep='\t')
            (loss + self.l1*l1_loss).backward()
            optimizer.step()

            if np.mean(losses[-10:]) < self.tol:
                break


class Jointclassifier(TextClassifier):
    def __init__(self, joint_model_pipeline, slot):
        self.pipeline = joint_model_pipeline
        self.slot = slot
        classifier = self.pipeline.named_steps['classifier']
        if slot.id not in classifier.labels:
            raise NotImplementedError()
        self.output_index = classifier.labels.index(self.slot.id)

    def predict_single(self, text: List[Dict[str, Any]]):
        if self.pipeline.predict_proba(text)[0, self.output_index] > THRESHOLD:
            return self.slot.true


def joint_intent_and_slot_classifier(slots: List[ClassifierSlot], models_folder):
    with open(os.path.join(models_folder, MODEL_FILE), 'rb') as f:
        joint_model_pipeline = pickle.load(f)
    for slot in slots:
        try:
            slot.classifier = Jointclassifier(joint_model_pipeline, slot)
        except NotImplementedError:
            pass

    return slots


def bce(predicted, true):
    return -(np.log(predicted)*true+np.log(1-predicted)*(1-true)).mean(0).sum()


def main(*args):
    parser = argparse.ArgumentParser(description='Train SVM and dump it')

    parser.add_argument('--folder', dest='model_folder', type=str, default=MODEL_FOLDER_DEFAULT,
                        help='The path for trained model')

    parser.add_argument('--data', dest='data_path', type=str, default='./generated_dataset.tsv',
                        help='The path of generated dataset')

    parser.add_argument('--slot_path', dest='slot_path', type=str, default="slots_definitions.tsv",
                        help='The path of file with slot definitions')

    parser.add_argument('--trash_intent', dest='trash_intent', type=str, default="sberdemo_no_intent.tsv.gz",
                        help='The path of file with trash intent examples')

    parser.add_argument('--cross_validation', action='store_true', default=False)

    params = vars(parser.parse_args(args))

    data_path = params['data_path']
    trash_dialogs_path = params['trash_intent']
    slot_defs_path = params['slot_path']

    # if there's no folder to save model
    # create folder
    if not os.path.exists(params['data_path']):
        os.mkdir(params['data_path'])

    assert os.path.exists(data_path), 'File "{}" not found'.format(data_path)

    # ------------ load slots ----------------------#

    pipe = create_pipe()
    slot_list = read_slots_from_tsv(pipeline=pipe, filename=slot_defs_path)
    slots = [[s.id, s] for s in slot_list if isinstance(s, ClassifierSlot)]
    slot_names = [name for name, slot in slots]
    print("Slot names: ", slot_names)

    # ------------ making train data ---------------#

    trash_data = sorted(set(pd.read_csv(trash_dialogs_path, compression='gzip', sep='\t', header=None).ix[:, 0]))[:560]

    data = pd.read_csv(data_path, sep='\t')
    intents = [i for i in data['intent'].unique() if pd.notnull(i)]
    intents_slots_map = dict(zip(chain(intents, slot_names), range(len(intents) + len(slot_names))))

    sents = []
    targets = np.zeros([len(data) + len(trash_data), max(intents_slots_map.values()) + 1])
    template_ids = []

    for i, (_, row) in enumerate(data.iterrows()):
        sents.append(row['request'])
        if pd.notnull(row['intent']):
            targets[i, intents_slots_map[row['intent']]] = 1
        template_ids.append(row['template_id'])
        for slot_name, slot in slots:
            slot_value = row[slot_name]
            if not pd.isnull(slot_value):
                targets[i, intents_slots_map[slot_name]] = 1

    sents.extend(trash_data)
    class_weights = targets.sum(axis=0).max() / targets.sum(axis=0)
    class_weights /= class_weights.sum()
    class_weights = np.zeros_like(class_weights) + 1
    print('Using following weights:')
    for i, name in enumerate(chain(intents, slot_names)):
        print(name.ljust(16, ' '), class_weights[i])

    print('normalizing data...')
    sents = [pipe.feed(s) for s in sents]
    print('done!')

    max_template_id = max(template_ids)
    template_ids.extend(range(max_template_id, max_template_id + len(trash_data)))

    # ------------ train a model ---------------#

    kf = GroupKFold(n_splits=5)

    all_predictions = []
    all_test_y = []

    filename = os.path.join(params['model_folder'], MODEL_FILE)

    if os.path.isfile(filename):
        os.unlink(filename)

    inv_intents_slots_map = {v: k for k, v in intents_slots_map.items()}
    labels = [inv_intents_slots_map[i] for i in range(len(inv_intents_slots_map))]
    base_estimator = MLPJointclassifier(tol=1e-2, hidden_layer_neurons=200, batch_size=32, l2=0.0001, l1=0.005,
                                        labels=labels)
    vectorizer = lambda: TfidfVectorizer(stop_words=COMMON_STOP_WORDS, ngram_range=(1, 2))

    if params['cross_validation']:
        for group_id, (train_index, test_index) in enumerate(kf.split(sents, targets, template_ids)):
            print('starting cross validation group {}'.format(group_id))
            train_sents = [sents[i] for i in train_index]
            train_y = targets[train_index]

            test_sents = [sents[i] for i in test_index]
            test_Y = targets[test_index]

            p = Pipeline([('to_string', StickSentence()),
                          ('feature_extractor', vectorizer()),
                          ('classifier', clone(base_estimator))])

            balancing_idx = joint_oversampling_coefs(train_y)
            balanced_sents = [train_sents[i] for i in balancing_idx]
            balanced_y = train_y[balancing_idx]

            p.fit(balanced_sents, balanced_y, class_weights=class_weights)

            predicted_y = p.predict_proba(test_sents)

            all_predictions.append(predicted_y)
            all_test_y.append(test_Y)


        all_test_y = np.vstack(all_test_y)
        all_predictions = np.vstack(all_predictions)

        print(base_estimator)

        for threshold in np.linspace(0.25, 0.45, 5):
            print('threshold', threshold)
            precision, recall, fbeta, support = precision_recall_fscore_support(all_test_y, all_predictions >= threshold)

            for name in chain(intents, slot_names):
                i = intents_slots_map[name]
                print(name.ljust(16, ' '), '{:.2f}'.format(precision[i]), '{:.2f}'.format(recall[i]),
                      '{:.2f}'.format(fbeta[i]), support[i], sep='\t')
            print()

    if os.path.isfile(filename):
        print('loading model from "{}"'.format(filename))
        with open(filename, 'rb') as f:
            p = pickle.load(f)
    else:
        joint_classifier = clone(base_estimator)
        p = Pipeline([('to_string', StickSentence()),
                      ('feature_extractor', vectorizer()),
                      ('classifier', joint_classifier)])

        balancing_idx = joint_oversampling_coefs(targets)
        balanced_x = [sents[i] for i in balancing_idx]
        balanced_y = targets[balancing_idx]
        p.fit(balanced_x, balanced_y)

        with open(filename, 'wb') as f:
            pickle.dump(p, f)

    fe = p.named_steps['feature_extractor']
    c = p.named_steps['classifier']
    voc = fe.vocabulary_
    idf = fe.idf_
    n = max(voc.values()) + 1
    words = []
    vec = np.zeros((len(voc), n))
    for i, (word, w_idx) in enumerate(voc.items()):
        words.append(word)
        # vec[i, w_idx] = -np.log(1+idf[w_idx])  # normalization turns to 1 anyway
        vec[i, w_idx] = 1
    importances = c.predict_proba(vec)

    for name, idx in intents_slots_map.items():
        print(name)
        for w, imp in sorted(zip(words, importances[:, idx]), key=itemgetter(1), reverse=True):
            if imp < 0.5:
                break
            print('{:.3f}'.format(imp), w, sep='\t')
        print()


if __name__ == '__main__':
    main(sys.argv[1:])
