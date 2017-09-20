import pandas as pd
from nlu import *
from intent_classifier import IntentClassifier
from collections import defaultdict
from sklearn.metrics import classification_report, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.externals import joblib
from svm_classifier_utlilities import oversample_data
from sklearn.svm import LinearSVC
from slots import read_slots_from_tsv, ClassifierSlot
import os
import argparse

import urllib.request

DUMP_DEFAULT = True
MODEL_FOLDER_DEFAULT = './models_nlu'
USE_CHAR_DEFAULT = False


def validate_train(model, X, y, groups, oversample=True, n_splits=5, use_chars=USE_CHAR_DEFAULT,
                   dump_name='any.model', dump=DUMP_DEFAULT, model_folder=MODEL_FOLDER_DEFAULT, metric=f1_score,
                   class_weights=None, verbose=False, num_importance = 20):

    kf = GroupKFold(n_splits=n_splits)
    all_y = []
    all_predicted = []
    for train_index, test_index in kf.split(X, y, groups):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        if oversample:
            X_tmp, y_tmp = oversample_data(X_train, y_train, verbose=verbose)
            model.train_model(X_tmp, y_tmp, use_chars=use_chars)
        else:
            model.train_model(X_train, y_train, use_chars=use_chars)
        pred = model.predict_batch(X_test)

        all_predicted.extend(pred)
        all_y.extend(y_test)

    print(">>> MODEL: ", dump_name)

    if metric is f1_score:
        result = metric(all_y, all_predicted, average=None)
    else:
        result = metric(all_y, all_predicted)

    if dump:
        if oversample:
            X_tmp, y_tmp = oversample_data(X, y, verbose=verbose)
            model.train_model(X_tmp, y_tmp, use_chars=use_chars)
        else:
            model.train_model(X, y, use_chars=use_chars)

        joblib.dump(model.model,
                    os.path.join(model_folder, dump_name))

        if isinstance(model.model.steps[2][1], LinearSVC):
            print("---Feature importance for {} ---".format(dump_name))
            coefs = model.model.steps[2][1].coef_[0]
            names = model.model.steps[1][1].words_vectorizer.get_feature_names()
            weights = sorted(list(zip(names, coefs)), key=lambda x: x[1], reverse=True)
            print("\n --- TOP {} most important --- \n".format(num_importance))
            for n, val in weights[:num_importance]:

                print("{}\t{}".format(n, np.round(val, 3)))
            print(print("\n --- TOP {} anti features --- \n".format(num_importance)))
            for n, val in weights[::-1][:num_importance]:
                print("{}\t{}".format(n, np.round(val, 3)))
        else:
            print("WHAT: ", type(model.model.steps[2][1]))
        print('==Model dumped==')
    print("classif_report:\n", classification_report(all_y, all_predicted))
    return result


def main(args=''):
    parser = argparse.ArgumentParser(description='Train SVM and dump it')

    parser.add_argument('--folder', dest='model_folder', type=str, default=MODEL_FOLDER_DEFAULT,
                        help='The path for trained model')

    parser.add_argument('--data', dest='data_path', type=str, default='./generated_dataset.tsv',
                        help='The path of generated dataset')

    parser.add_argument('--dump', dest='dump', action='store_true', default=DUMP_DEFAULT,
                        help='Use flag to dump trained svm')

    parser.add_argument('--oversample', dest='oversample', action='store_true', default=False,
                        help='Use flag to test and dump models with oversample')

    parser.add_argument('--use_char', dest='use_char', action='store_true', default=USE_CHAR_DEFAULT,
                        help='Use flag to use char features in svm')

    parser.add_argument('--slot_path', dest='slot_path', type=str, default="slots_definitions.tsv",
                        help='The path of file with slot definitions')

    parser.add_argument('--trash_intent', dest='trash_intent', type=str, default="sberdemo_no_intent.tsv.gz",
                        help='The path of file with trash intent examples')

    parser.add_argument('--slot_train', dest='slot_train', action='store_true', default=False,
                        help="Use flag to train slots' svms ")

    parser.add_argument('--intent_train', dest='intent_train', action='store_true', default=False,
                        help="Use flag to train intent multiclass svm")

    parser.add_argument('--num_importance', dest='num_importance', type=int, default=20,
                        help="How many samples to show in feature importance")


    args = parser.parse_args(args)
    params = vars(args)

    MODEL_FOLDER = params['model_folder']
    DUMP = params['dump']  # True to save model for each slot
    DATA_PATH = params['data_path']
    NO_INTENT = params['trash_intent']
    OVERSAMPLE = params['oversample']
    SLOT_PATH = params['slot_path']
    USE_CHAR = params['use_char']
    INTENT_TRAIN = params['intent_train']
    SLOT_TRAIN = params['slot_train']
    NUM_IMPORTANCE = params['num_importance']

    # just checking:
    print("Current configuration:\n")
    print(params)

    # if there's no folder to save model
    # create folder
    if not os.path.exists(MODEL_FOLDER):
        os.mkdir(MODEL_FOLDER)

    # if there's no file with generated data
    # generate data
    if not os.path.exists(DATA_PATH):
        os.system('python generate_from_templates.py')

    # ------------ load slots ----------------------#

    pipe = create_pipe()
    slot_list = read_slots_from_tsv(pipeline=pipe, filename=SLOT_PATH)
    slot_names = [s.id for s in slot_list if isinstance(s, ClassifierSlot)]
    print("Slot names: ", slot_names)

    # ------------ making train data ---------------#

    if not os.path.isfile(NO_INTENT):
        url = 'http://share.ipavlov.mipt.ru:8080/repository/datasets/' + os.path.basename(NO_INTENT)
        try:
            urllib.request.urlretrieve(url, NO_INTENT)
        except:
            pass

    trash_data = list(set(pd.read_csv(NO_INTENT, compression='gzip',  sep='\t', header=None).ix[:, 0]))
    data = pd.read_csv(DATA_PATH, sep='\t')
    sents = []
    targets = defaultdict(list)

    for i, row in data.iterrows():
        sents.append(row['request'])

        # add targets
        for slot in slot_names:
            targets[slot].append(not pd.isnull(row[slot]))

    y_intents = list(data['intent'])
    X = []
    for s in sents:
        X.append([w['normal'] for w in pipe.feed(s)])

    trash_sents = trash_data[:len(y_intents)]
    X_intents = list(X)
    for s in trash_data[:len(y_intents)]:
        X_intents.append([w['normal'] for w in pipe.feed(s)])
    X_intents = np.array(X_intents)

    y_intents = np.array(y_intents + ['no_intent'] * len(trash_sents))

    # ---------------- validate & dump --------------#



    if INTENT_TRAIN:
        intent_clf = IntentClassifier(labels_list=y_intents)
        print("intent_clf.string2idx: ", intent_clf.string2idx)
        print("\n-------\n")
        y_intents_idx = np.array([intent_clf.string2idx[t] for t in y_intents])
        if DUMP:
            joblib.dump(intent_clf.string2idx, os.path.join(MODEL_FOLDER, "string2idx_dict.model"))
        tmp_max = max(data['template_id'])
        tmp_groups = list(data['template_id']) + list(range(tmp_max + 1, tmp_max + len(trash_sents) + 1))
        result = validate_train(intent_clf, X_intents, y_intents_idx,
                                groups=tmp_groups,
                                oversample=OVERSAMPLE,
                                metric=f1_score,
                                n_splits=8,
                                dump_name="IntentClassifier.model",
                                num_importance=NUM_IMPORTANCE,
                                class_weights=None)
        print("INTENT CLF: cv mean f1 score: {}".format(result))

        print('--------------')

    if SLOT_TRAIN:
        for slot in slot_list:
            if slot.id not in slot_names:
                continue
            result = validate_train(model=slot, X=np.array(X), y=np.array(targets[slot.id]),
                                    groups=data['template_id'],
                                    oversample=OVERSAMPLE,
                                    n_splits=8,
                                    metric=f1_score,
                                    num_importance=NUM_IMPORTANCE,
                                    dump_name="{}.model".format(slot.id))
            print("For slot: {} cv mean f1 score: {}".format(slot.id, result))
            print('--------------')


if __name__ == '__main__':
    main()
