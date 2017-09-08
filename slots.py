import csv
import os

import sys
from itertools import chain
from typing import Dict, List, Any, Union

from fuzzywuzzy import fuzz
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from slots_classifier_utlilities import FeatureExtractor


class DictionarySlot:
    def __init__(self, slot_id: str, ask_sentence: str, generative_dict: Dict[str, str],
                 nongenerative_dict: Dict[str, str]):
        self.id = slot_id
        self.ask_sentence = ask_sentence
        self.gen_dict = generative_dict
        self.nongen_dict = nongenerative_dict
        self.threshold = 80

        self.filters = {
            'any': lambda x, _: True,
            'eq': lambda x, y: x == y,
            'not_eq': lambda x, y: x != y,
            'true': lambda x, _: bool(x),
            'false': lambda x, _: not bool(x)
        }

    def infer_from_compositional_request(self, text):
        return self._infer(text)

    def infer_from_single_slot(self, text):
        return self._infer(text)

    def _normal_value(self, text: str) -> str:
        return self.gen_dict.get(text, self.nongen_dict.get(text, ''))

    def _infer(self, text: List[Dict[str, Any]]) -> Union[str, None]:
        str_text = ' '.join(w['_text'] for w in text)
        best_score = 0
        best_match = None
        for v in chain(self.gen_dict, self.nongen_dict):
            score = fuzz.partial_ratio(v, str_text)
            if score > best_score:
                best_score = score
                best_match = v

        # works poorly for unknown reasons
        # print(process.extractBests(str_text, choices=[str(x) for x in chain(self.gen_dict, self.nongen_dict)], scorer=fuzz.partial_ratio))
        if best_score >= self.threshold:
            return self._normal_value(best_match)
        return None

    def __repr__(self):
        return '{}(name={}, len(dict)={})'.format(self.__class__.__name__, self.id, len(self.gen_dict))

    def filter(self, value: str) -> bool:
        raise NotImplemented()

    def ask(self) -> str:
        return self.ask_sentence


class ClassifierSlot(DictionarySlot):
    def __init__(self, slot_id: str, ask_sentence: str, generative_dict: Dict[str, str],
                 nongenerative_dict: Dict[str, str]):
        super().__init__(slot_id, ask_sentence, generative_dict, nongenerative_dict)
        self.model = None

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise Exception("Model path: '{}' doesnt exist".format(model_path))
        self.model = joblib.load(model_path)

    def train_model(self, X, y, use_chars=False):
        """
        :param X: iterable with strings
        :param y: target binary labels
        :param use_chars: True if use char features
        :return: None

        """
        feat_generator = FeatureExtractor(use_chars=use_chars)
        clf = SVC()
        self.model = Pipeline([('feature_extractor', feat_generator), ('svc', clf)])
        self.model.fit(X, y)

    def infer_from_compositional_request(self, text):
        """
        :param text: just string
        :return:
        """
        if self.model is None:
            raise NotImplementedError("No model specified!")

        label = self.model.predict(text)[0]
        return bool(label)


class CompositionalSlot(DictionarySlot):
    def __init__(self, slot_id: str, ask_sentence: str, generative_dict: Dict[str, str],
                 nongenerative_dict: Dict[str, str]):
        super().__init__(slot_id, ask_sentence, generative_dict, nongenerative_dict)


class TomitaSlot(DictionarySlot):
    def __init__(self, slot_id: str, ask_sentence: str, generative_dict: Dict[str, str],
                 nongenerative_dict: Dict[str, str]):
        super().__init__(slot_id, ask_sentence, generative_dict, nongenerative_dict)


class GeoSlot(DictionarySlot):
    def __init__(self, slot_id: str, ask_sentence: str, generative_dict: Dict[str, str],
                 nongenerative_dict: Dict[str, str]):
        super().__init__(slot_id, ask_sentence, generative_dict, nongenerative_dict)


def read_slots_from_tsv(filename=None):
    D = '\t'
    if filename is None:
        filename = 'slots_definitions.tsv'
    with open(filename) as f:
        csv_rows = csv.reader(f, delimiter=D, quotechar='"')
        slot_name = None
        slot_class = None
        info_question = None
        generative_slot_values = {}
        nongenerative_slot_values = {}

        result_slots = []
        for row in csv_rows:
            if slot_name is None:
                slot_name, slot_class, *args = row[0].split()[0].split('.')
                info_question = row[1].strip()
            elif ''.join(row):
                nongenerative_syns = ''
                generative_syns = ''
                if len(row) == 1:
                    normal_name = row[0]
                elif len(row) == 2:
                    normal_name, generative_syns = row
                elif len(row) == 3:
                    normal_name, generative_syns, nongenerative_syns = row
                else:
                    raise Exception()

                if generative_syns:
                    generative_syns = generative_syns.replace(', ', ',').replace('“', '').replace('”', '').replace('"',
                                                                                                                   '').split(
                        ',')
                else:
                    generative_syns = []

                if nongenerative_syns:
                    nongenerative_syns = nongenerative_syns.replace(', ', ',').replace('“', '').replace('”',
                                                                                                        '').replace('"',
                                                                                                                    '').split(
                        ',')
                else:
                    nongenerative_syns = []

                if nongenerative_syns and generative_syns:
                    assert not (set(nongenerative_syns).intersection(set(generative_syns))), [nongenerative_syns,
                                                                                              generative_syns]

                for s in nongenerative_syns:
                    nongenerative_slot_values[s] = normal_name

                generative_slot_values[normal_name] = normal_name
                for s in generative_syns:
                    generative_slot_values[s] = normal_name
            else:
                SlotClass = getattr(sys.modules[__name__], slot_class)
                slot = SlotClass(slot_name, info_question, generative_slot_values, nongenerative_slot_values)
                result_slots.append(slot)

                slot_name = None
                generative_slot_values = {}
                nongenerative_slot_values = {}
        if slot_name:
            SlotClass = getattr(sys.modules[__name__], slot_class)
            slot = SlotClass(slot_name, info_question, generative_slot_values, nongenerative_slot_values)
            result_slots.append(slot)

    return result_slots


def read_slots_serialized(folder):
    """
    Read slots from tsv and load saved svm models

    :param folder: path to folder with models
    :return: array of slots

    """
    slots_array = read_slots_from_tsv()

    for s in slots_array:
        s.load_model(os.path.join(folder, s.id + '.model'))
    return slots_array