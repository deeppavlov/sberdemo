import os
from sklearn.linear_model import LogisticRegression
from svm_classifier_utlilities import SentenceClassifier


class IntentClassifier(SentenceClassifier):
    def __init__(self, base_clf=LogisticRegression(penalty='l1', C=10),
                 stop_words=None, use_chars=False, labels_list=None, folder=None):

        self.model_name = "IntentClassifier.model"
        if folder is not None:
            model_path = os.path.join(folder, self.model_name)
        else:
            model_path = None

        super().__init__(model_path=model_path, base_clf=base_clf,
                         use_chars=use_chars, stop_words=stop_words,
                         labels_list=labels_list)

    def dump_model(self, folder):
        super().dump_model(os.path.join(folder, self.model_name))

