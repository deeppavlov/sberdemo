import pandas as pd
from nlu import *
from intent_classifier import IntentClassifier
from collections import defaultdict
from sklearn.metrics import classification_report, f1_score, precision_recall_fscore_support
from sklearn.model_selection import GroupKFold
from sklearn.externals import joblib
from svm_classifier_utlilities import oversample_data
from sklearn.svm import LinearSVC
from slots import read_slots_from_tsv, ClassifierSlot
import os
import argparse

import csv
import datetime

DUMP_DEFAULT = True
MODEL_FOLDER_DEFAULT = './models_nlu'
USE_CHAR_DEFAULT = False
STOP_WORDS_INTENT = ["здравствовать", "добрый", "день", "для", "хотеть", "нужный"]
STOP_WORDS_SLOTS = {"online_reserving":['ли', 'открыть', "счет", "здравствовать", "возможно",
                                        "через", "приходить", "банк", "как", "через", "без", "день"],


                    "show_docs":["открытие", "счет", "здравствовать", "это", "при", "для",
                                 "какой", "нужный", "необходимый", "по", "сбербанк", "хотеть", "открыть", "бы"],


                    "cost_of_service":["по", "здравствовать", "рассказать", "открытие", "день",
                                       "хотеть", "открыть", "сказать", "какой", "бы", "счет", "для", "мочь", "счет", "сбер"],

                    "show_phone":["график", "пожалуйста", "здравствовать", "можно", "работать",
                                  "это", "офис", "строгино", "банк", "работать", "сказать", "ленинский",
                                  "отделение", "банка" "ряд", "чертановский", "где", "ближний", "банк", "день"],


                    "show_schedule":["телефон","здравствовать", "работа", "ряд", "офис"],


                    "search_vsp":["здравствовать"],

                    "not_first":["открыть", "что", "заявление", "мочь", "здравствовать", "необходимый", "комплект документ"]}
BASE_CLF = LinearSVC(C=1)
BASE_CLF_INTENT = BASE_CLF
BASE_CLF_SLOTS = BASE_CLF


def write_to_tsv(fname, headers, data):
    exists = os.path.isfile(fname)

    with open(fname, 'a') as tsv_file:
        writer = csv.writer(tsv_file, delimiter='\t', lineterminator='\n')
        if not exists:
            writer.writerow(headers)
        writer.writerow(data)


def ensure_log_folder():
    folder = os.path.join('.', 'logs', 'models')
    if not os.path.isdir(folder):
        os.mkdir(path=folder)
    return folder


def log_results(y_true, y_predicted, model_name, model):
    path = os.path.join(ensure_log_folder(), '{}.tsv'.format(model_name))

    p, r, f1, s = precision_recall_fscore_support(y_true, y_predicted)
    labels = sorted(list(set(y_true)))
    labels = [str(model.idx2string[_]) for _ in labels]

    headers = ['Time', 'Description']
    data = [datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'), model.get_description()]
    for i in range(len(labels)):
        headers += [_.format(labels[i]) for _ in ['{}_precision', '{}_recall', '{}_f1', '{}_support']]
        data += [p[i], r[i], f1[i], s[i]]

    write_to_tsv(path, headers, data)


def log_slots_aggr_results(results):
    headers, data = [list(_) for _ in zip(*results)]
    path = os.path.join(ensure_log_folder(), 'slots_aggr.tsv')
    headers.insert(0, ['Time'])
    data.insert(0, datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
    write_to_tsv(path, headers, data)


def validate_train(model, X, y, groups, oversample=True, n_splits=5, use_chars=USE_CHAR_DEFAULT,
                   dump_name='any.model', dump=DUMP_DEFAULT, model_folder=MODEL_FOLDER_DEFAULT, metric=f1_score,
                   class_weights=None, verbose=False, stop_words=None, num_importance=20, base_clf=BASE_CLF):
    kf = GroupKFold(n_splits=n_splits)
    all_y = []
    all_predicted = []
    for train_index, test_index in kf.split(X, y, groups):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        if oversample:
            X_tmp, y_tmp = oversample_data(X_train, y_train, verbose=verbose)
            model.train_model(X_tmp, y_tmp, use_chars=use_chars, stop_words=stop_words, base_clf=base_clf)
        else:
            model.train_model(X_train, y_train, use_chars=use_chars, stop_words=stop_words, base_clf=base_clf)
        pred = model.predict_batch(X_test)

        all_predicted.extend(pred)
        all_y.extend(y_test)

    print(">>> MODEL: ", dump_name)
    print("Params:", model.get_description())
    all_y = model.encode2idx(all_y)

    if metric is f1_score:
        result = metric(all_y, all_predicted, average=None)
    else:
        result = metric(all_y, all_predicted)

    if dump:
        if oversample:
            X_tmp, y_tmp = oversample_data(X, y, verbose=verbose)
            model.train_model(X_tmp, y_tmp, use_chars=use_chars, stop_words=stop_words, base_clf=base_clf)
        else:
            model.train_model(X, y, use_chars=use_chars, stop_words=stop_words, base_clf=base_clf)

        print("FEATURE_IMPORTANCE")

        importances = model.get_feature_importance()
        labels = model.get_labels()
        print("=== labels {} ===".format(labels))
        for imp_line, label in zip(importances, labels):
            print("\nLABEL: ", label)
            print("*"*20)
            print("\n --- TOP {} most important --- \n".format(num_importance))
            for n, val in imp_line[:num_importance]:
                print("{}\t{}".format(n, np.round(val, 3)))

            print("\n --- TOP {} anti features --- \n".format(num_importance))
            for n, val in imp_line[::-1][:num_importance]:
                print("{}\t{}".format(n, np.round(val, 3)))

        model.dump_model(os.path.join(model_folder, dump_name))
        print("== MODEL DUMPED ==")

    print("classif_report:\n", classification_report(all_y, all_predicted))
    log_results(all_y, all_predicted, dump_name or model.model_name, model)
    return result


def main(args=None):
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

    parser.add_argument('--num_importance', dest='num_importance', type=int, default=30,
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

    trash_data = list(set(pd.read_csv(NO_INTENT, compression='gzip', sep='\t', header=None).ix[:, 0]))
    data = pd.read_csv(DATA_PATH, sep='\t')
    sents = []
    targets = defaultdict(list)

    for i, row in data.iterrows():
        sents.append(row['request'])

        # add targets
        for name, slot in slots:
            label = '_' if pd.isnull(row[name]) else slot.true
            targets[name].append(label)

    y_intents = list(data['intent'])
    X = [pipe.feed(s) for s in sents]

    trash_sents = trash_data[:len(y_intents)]
    X_intents = list(X) + [pipe.feed(s) for s in trash_data[:len(y_intents)]]

    X_intents = np.array(X_intents)

    y_intents = np.array(y_intents + ['no_intent'] * len(trash_sents))
    tmp_max = max(data['template_id'])
    tmp_groups = list(data['template_id']) + list(range(tmp_max + 1, tmp_max + len(trash_sents) + 1))

    # ---------------- validate & dump --------------#

    if INTENT_TRAIN:
        intent_clf = IntentClassifier(labels_list=y_intents)
        print("intent_clf.string2idx: ", intent_clf.get_labels())
        print("\n-------\n")
        result = validate_train(intent_clf, X_intents, y_intents,
                                groups=tmp_groups,
                                oversample=OVERSAMPLE,
                                metric=f1_score,
                                n_splits=8,
                                dump_name="",
                                num_importance=NUM_IMPORTANCE,
                                class_weights=None,
                                use_chars=USE_CHAR,
                                stop_words=STOP_WORDS_INTENT,
                                base_clf=BASE_CLF_INTENT)
        print("INTENT CLF: cv mean f1 score: {}".format(result))

        print('--------------\n\n')

    if SLOT_TRAIN:
        results = []
        for slot in slot_list:
            if slot.id not in slot_names:
                continue

            model_name = "{}.model".format(slot.id)
            result = validate_train(model=slot, X=X_intents, y=np.array(targets[slot.id] + ['_'] * len(trash_sents)),
                                    groups=tmp_groups,
                                    oversample=OVERSAMPLE,
                                    n_splits=8,
                                    metric=f1_score,
                                    num_importance=NUM_IMPORTANCE,
                                    dump_name=model_name,
                                    use_chars=USE_CHAR,
                                    stop_words=STOP_WORDS_SLOTS.get(slot.id,None),
                                    base_clf=BASE_CLF_SLOTS)
            print("For slot: {} cv mean f1 score: {}".format(slot.id, result))
            results.append((model_name, sum(result)/len(result)))

            print('--------------\n\n')

        if results:
            log_slots_aggr_results(results)


if __name__ == '__main__':
    main()
