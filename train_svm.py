import pandas as pd
from nlu import *
from collections import defaultdict
from sklearn.metrics import confusion_matrix, roc_auc_score, f1_score
from sklearn.model_selection import GroupKFold
from sklearn.externals import joblib
from slots_classifier_utlilities import oversample
from slots import *
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

args = parser.parse_args()
params = vars(args)

MODEL_FOLDER = params['model_folder']
DUMP = params['dump']  # True to save model for each slot
SAVE_PIC = params['save_pic']
DATA_PATH = params['data_path']
OVERSAMPLE = params['oversample']
SLOT_PATH = params['slot_path']
USE_CHAR = params['use_char']

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
print(slot_names)

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

# ---------------- validate & dump --------------#


# leave one out
# kf = GroupKFold(n_splits=len(data['template_id'].unique()))
kf = GroupKFold(n_splits=5)
scores = []
all_y = []
all_predicted = []
groups = data['template_id']
for slot in slot_list:
    if slot.id not in slot_names:
        continue

    print("SLOT: ", slot.id)
    y = np.array(targets[slot.id])
    for train_index, test_index in kf.split(X, y, groups):
        X_train, y_train = X[train_index], y[train_index]
        X_test, y_test = X[test_index], y[test_index]
        if OVERSAMPLE:
            X_tmp, y_tmp = oversample(X_train, y_train, verbose=True)
            slot.train_model(X_tmp, y_tmp, use_chars=USE_CHAR)
        else:
            slot.train_model(X_train, y_train, use_chars=USE_CHAR)
        pred = slot.infer_from_compositional_batch(X_test)

        all_y.extend(y_test)
        all_predicted.extend(pred)

    result = f1_score(all_y, all_predicted)
    print("For slot: {} cv mean f1 score: {}".format(slot.id, result))
    print('--------------')
    if DUMP:
        if OVERSAMPLE:
            X_tmp, y_tmp = oversample(X, y)
            slot.train_model(X_tmp, y_tmp, use_chars=USE_CHAR)
        else:
            slot.train_model(X, y, use_chars=USE_CHAR)

        joblib.dump(slot.model,
                    os.path.join(MODEL_FOLDER, '{}.model'.format(slot.id)))

        print('==Model dumped==')