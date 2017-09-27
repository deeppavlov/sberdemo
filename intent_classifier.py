import os
from sklearn.linear_model import LogisticRegression
from svm_classifier_utlilities import SentenceClassifier


class IntentClassifier(SentenceClassifier):
    def __init__(self, base_clf, stop_words=None, use_chars=False, labels_list=None, folder=None):

        model_name = "IntentClassifier.model"

        if folder is not None:
            model_path = os.path.join(folder, model_name)
        else:
            model_path = None

        super().__init__(base_clf, model_path=model_path,
                         use_chars=use_chars, stop_words=stop_words,
                         labels_list=labels_list, model_name=model_name)


