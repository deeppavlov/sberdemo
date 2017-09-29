from itertools import chain

import pandas as pd
from scipy.optimize import fmin
from sklearn.linear_model import ElasticNet
from sklearn.metrics import f1_score, precision_recall_fscore_support

from nlu import *
from sklearn.model_selection import GroupKFold
from slots import read_slots_from_tsv, ClassifierSlot

import os
import argparse

import torch
import torch.tensor as T
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

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
        # if False:
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


class MLPJointClassifier(BaseEstimator):
    def __init__(self, hidden_layer_neurons=40, l1=0.0, l2=0.0, tol=1e-3, batch_size=1):
        super().__init__()
        self.hidden_layer_neurons = hidden_layer_neurons
        self.l1 = l1
        self.l2 = l2
        self.tol = tol
        self.batch_size = batch_size



    def predict(self, X):
        return self.predict_proba(X) > 0.5

    def predict_proba(self, X):
        return self.model.forward(Variable(torch.FloatTensor(X))).data.numpy()

    def fit(self, X: np.ndarray, Y: np.ndarray, class_weights: np.ndarray):
        assert X.ndim == 2
        assert Y.ndim == 2
        W1 = nn.Linear(X.shape[1], self.hidden_layer_neurons)
        W2 = nn.Linear(self.hidden_layer_neurons, Y.shape[1])

        X = Variable(torch.FloatTensor(X))
        Y = Variable(torch.FloatTensor(Y))
        W = Variable(torch.FloatTensor(class_weights[np.newaxis]))

        self.model = nn.Sequential(W1, nn.ReLU(), W2, nn.Sigmoid())

        criterion = nn.BCELoss()

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


def main(args=None):
    parser = argparse.ArgumentParser(description='Train SVM and dump it')

    parser.add_argument('--folder', dest='model_folder', type=str, default=MODEL_FOLDER_DEFAULT,
                        help='The path for trained model')

    parser.add_argument('--data', dest='data_path', type=str, default='./generated_dataset.tsv',
                        help='The path of generated dataset')

    parser.add_argument('--slot_path', dest='slot_path', type=str, default="slots_definitions.tsv",
                        help='The path of file with slot definitions')

    parser.add_argument('--trash_intent', dest='trash_intent', type=str, default="sberdemo_no_intent.tsv.gz",
                        help='The path of file with trash intent examples')

    args = parser.parse_args(args)
    params = vars(args)

    MODEL_FOLDER = params['model_folder']
    DATA_PATH = params['data_path']
    NO_INTENT = params['trash_intent']
    SLOT_PATH = params['slot_path']

    # if there's no folder to save model
    # create folder
    if not os.path.exists(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)

    assert os.path.exists(DATA_PATH), 'File "{}" not found'.format(DATA_PATH)

    # ------------ load slots ----------------------#

    pipe = create_pipe()
    slot_list = read_slots_from_tsv(pipeline=pipe, filename=SLOT_PATH)
    slots = [[s.id, s] for s in slot_list if isinstance(s, ClassifierSlot)]
    slot_names = [name for name, slot in slots]
    print("Slot names: ", slot_names)

    # ------------ making train data ---------------#

    trash_data = list(set(pd.read_csv(NO_INTENT, compression='gzip', sep='\t', header=None).ix[:, 0]))[:560]

    data = pd.read_csv(DATA_PATH, sep='\t')
    intents = data['intent'].unique()
    intents_slots_map = dict(zip(chain(intents, slot_names), range(len(intents) + len(slot_names))))

    sents = []
    targets = np.zeros([len(data) + len(trash_data), max(intents_slots_map.values()) + 1])
    template_ids = []

    for i, (_, row) in enumerate(data.iterrows()):
        sents.append(row['request'])
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
    sents = [' '.join(w['normal'] for w in pipe.feed(s)) for s in sents]
    print('done!')

    max_template_id = max(template_ids)
    template_ids.extend(range(max_template_id, max_template_id + len(trash_data)))


    # ------------ train a model ---------------#

    kf = GroupKFold(n_splits=8)

    all_predictions = []
    all_test_y = []

    base_estimator = MLPJointClassifier(tol=1e-1, hidden_layer_neurons=200, batch_size=32, l2=0.0001, l1=0.008)

    if need_cross_validation:
        for group_id, (train_index, test_index) in enumerate(kf.split(sents, targets, template_ids)):
            print('starting cross validation group {}'.format(group_id))
            train_sents = [sents[i] for i in train_index]
            train_y = targets[train_index]

            test_sents = [sents[i] for i in test_index]
            test_Y = targets[test_index]

            joint_classifier = clone(base_estimator)
            fe = FeatureExtractor(stop_words=COMMON_STOP_WORDS)
            train_X = fe.fit_transform(train_sents)

            balancing_idx = joint_oversampling_coefs(train_y)
            balanced_x = train_X[balancing_idx]
            balanced_y = train_y[balancing_idx]

            test_X = fe.transform(test_sents)

            joint_classifier.fit(balanced_x, balanced_y, class_weights=class_weights)

            predicted_y = joint_classifier.predict_proba(test_X)

            all_predictions.append(predicted_y)
            all_test_y.append(test_Y)


        all_test_y = np.vstack(all_test_y)
        all_predictions = np.vstack(all_predictions)

        print(joint_classifier)

        for threshold in np.linspace(0.25, 0.45, 5):
            print('threshold', threshold)
            precision, recall, fbeta, support = precision_recall_fscore_support(all_test_y, all_predictions >= threshold)

            for name in chain(intents, slot_names):
                i = intents_slots_map[name]
                print(name.ljust(16, ' '), '{:.2f}'.format(precision[i]), '{:.2f}'.format(recall[i]),
                      '{:.2f}'.format(fbeta[i]), support[i], sep='\t')
            print()

    joint_classifier = clone(base_estimator)
    fe = FeatureExtractor(stop_words=COMMON_STOP_WORDS)
    train_X = fe.fit_transform(sents)
    #
    # balancing_idx = joint_oversampling_coefs(targets)
    # balanced_x = train_X[balancing_idx]
    # balanced_y = targets[balancing_idx]
    # joint_classifier.fit(balanced_x, balanced_y, class_weights=class_weights)
    #
    # joint_classifier.save('nn.model')
    #
    # predicted_y = joint_classifier.predict_proba(train_X)

    predicted_y = np.load('predictions.ndarr.npy')

    loaded_model = torch.load('nn.model')

    loaded_predicted_y = loaded_model.predict_proba(train_X)

    np.allclose(loaded_predicted_y, predicted_y)

    # np.save('predictions.ndarr', predicted_y)



if __name__ == '__main__':
    main()
