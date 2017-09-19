import os
from sklearn.externals import joblib
from svm_classifier_utlilities import FeatureExtractor, StickSentence
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC
from typing import Dict, List, Any


class IntentClassifier():
    def __init__(self, labels_list=None, folder=None):
        self.model = None

        if labels_list is not None:
            # sort strings to fix their labels
            self.label_list = sorted(list(set(labels_list)), key=str.lower)
            self.string2idx = {t: i for i, t in enumerate(self.label_list)}
            self.idx2string = {v: k for k, v in self.string2idx.items()}
        else:
            self.string2idx = dict()
            self.idx2string = dict()

        if folder is not None:
            self.load_model(folder)

    def train_model(self, X: List[List[Dict[str, Any]]], y: List[str], use_chars=False):
        """
        :param X: List[List[Dict[str, Any]]] -- pipelinePreprocess output
        :param y: List[str]
        :param use_chars: True if use char features
        :return: None

        """
        feat_generator = FeatureExtractor(use_chars=use_chars)

        if isinstance(y[0], str):
            y_idx = [self.string2idx[s] for s in y]
        else:
            y_idx = y

        clf = LinearSVC()

        sticker_sent = StickSentence()
        self.model = Pipeline([("sticker_sent", sticker_sent), ('feature_extractor', feat_generator), ('svc', clf)])
        self.model.fit(X, y_idx)

    def predict_single(self, text: List[Dict[str, Any]]):
        if self.model is None:
            raise NotImplementedError("No model specified!")
        label = self.model.predict(text)[0]
        return self.idx2string[label]

    def predict_batch(self, list_texts: List[List[Dict[str, Any]]]):
        if self.model is None:
            raise NotImplementedError("No model specified!")
        labels = self.model.predict(list_texts)
        return labels

    def load_model(self, folder):
        model_path = os.path.join(folder, 'IntentClassifier.model')
        dict_path = os.path.join(folder, 'string2idx_dict.model')
        if not os.path.exists(model_path):
            raise Exception("Model path: '{}' doesnt exist".format(model_path))
        if not os.path.exists(dict_path):
            raise Exception("Dict path: '{}' doesnt exist".format(dict_path))
        self.model = joblib.load(model_path)
        self.string2idx = joblib.load(dict_path)
        self.idx2string = {v: k for k, v in self.string2idx.items()}
        print("Model loaded\n")
