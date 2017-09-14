import pandas as pd
from nlu import *
from intent_classifier import IntentClassifier
from collections import defaultdict
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score, accuracy_score
from sklearn.model_selection import GroupKFold
from sklearn.externals import joblib
from svm_classifier_utlilities import oversample_data
from slots import read_slots_from_tsv, ClassifierSlot
from numpy.random import RandomState
import os
import argparse

parser = argparse.ArgumentParser(description='Train SVM and dump it')

parser.add_argument('--folder', dest='model_folder', type=str, default='./models_nlu',
                    help='The path for trained model')

parser.add_argument('--data', dest='data_path', type=str, default='./generated_dataset.tsv',
                    help='The path of generated dataset')

parser.add_argument('--dump', dest='dump', action='store_true', default=True,
                    help='Use flag to dump trained svm')

parser.add_argument('--oversample', dest='oversample', action='store_false', default=True,
                    help='Use flag to test and dump models !without! oversample; defaule -- use oversample;')

parser.add_argument('--pic', dest='save_pic', action='store_true', default=True,
                    help='Use flag to save TSNE')

parser.add_argument('--use_char', dest='use_char', action='store_true', default=False,
                    help='Use flag to use char features in svm')

parser.add_argument('--slot_path', dest='slot_path', type=str, default="slots_definitions.tsv",
                    help='The path of file with slot definitions')

parser.add_argument('--slot_train', dest='slot_train', action='store_true', default=True,
                    help="Use flag to train slots' svms ")

parser.add_argument('--intent_train', dest='intent_train', action='store_true', default=True,
                    help="Use flag to train intent multiclass svm")

args = parser.parse_args()
params = vars(args)

MODEL_FOLDER = params['model_folder']
DUMP = params['dump']  # True to save model for each slot
SAVE_PIC = params['save_pic']
DATA_PATH = params['data_path']
OVERSAMPLE = params['oversample']
SLOT_PATH = params['slot_path']
USE_CHAR = params['use_char']
INTENT_TRAIN = params['intent_train']
SLOT_TRAIN = params['slot_train']

random_state = RandomState(23)

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

data = pd.read_csv(DATA_PATH, sep='\t')
sents = []
targets = defaultdict(list)

for i, row in data.iterrows():
    sents.append(row['request'])

    # add targets
    for slot in slot_names:
        targets[slot].append(not pd.isnull(row[slot]))

X = np.array([pipe.feed(sent) for sent in sents])  # list of list of dicts;
y_intents = np.array(list(data['intent']))


# ---------------- validate & dump --------------#

def validate_train(model, X, y, oversample=OVERSAMPLE, n_splits=5, use_chars=USE_CHAR,
                   dump_name='any.model', dump=DUMP, model_folder=MODEL_FOLDER, metric=f1_score, verbose=True):
    kf = GroupKFold(n_splits=n_splits)
    groups = data['template_id']
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
        # test_score = metric(y_test, pred)

        # print("     >>pred!: ", pred)
        # print("     >>true!: ", y_test)
        # print(">> ", test_score)
        # print("     test_len: ", len(y_test))

    result = metric(all_y, all_predicted)
    if dump:
        if oversample:
            X_tmp, y_tmp = oversample_data(X, y, verbose=verbose)
            model.train_model(X_tmp, y_tmp, use_chars=use_chars)
        else:
            model.train_model(X, y, use_chars=use_chars)

        joblib.dump(model.model,
                    os.path.join(model_folder, dump_name))
        print('==Model dumped==')
    return result


if INTENT_TRAIN:
    intent_clf = IntentClassifier(labels_list=y_intents)
    print("intent_clf.string2idx: ", intent_clf.string2idx)
    print("\n-------\n")
    y_intents_idx = np.array([intent_clf.string2idx[t] for t in y_intents])
    if DUMP:
        joblib.dump(intent_clf.string2idx, os.path.join(MODEL_FOLDER, "string2idx_dict.model"))

    result = validate_train(intent_clf, X, y_intents_idx, oversample=True, metric=f1_score,
                            n_splits=8, dump_name="IntentClassifier.model", verbose=False)
    print("INTENT CLF: cv mean f1 score: {}".format(result))
    print('--------------')

if SLOT_TRAIN:
    for slot in slot_list:
        if slot.id not in slot_names:
            continue

        print("SLOT: ", slot.id)
        result = validate_train(slot, X, np.array(targets[slot.id]), n_splits=8, metric=f1_score,
                                dump_name="{}.model".format(slot.id), verbose=False)
        print("For slot: {} cv mean f1 score: {}".format(slot.id, result))
        print('--------------')
