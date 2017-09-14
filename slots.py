import csv
import os
import random

import sys
from collections import defaultdict
from itertools import chain
from typing import Dict, List, Any, Union

from fuzzywuzzy import fuzz
from sklearn.externals import joblib
from sklearn.pipeline import Pipeline
from sklearn.svm import SVC

from svm_classifier_utlilities import FeatureExtractor
from svm_classifier_utlilities import StickSentence
from tomita.tomita import Tomita


class DictionarySlot:
    def __init__(self, slot_id: str, ask_sentence: str, generative_dict: Dict[str, str],
                 nongenerative_dict: Dict[str, str], values_order: List[str], prev_created_slots: List, *args):
        self.id = slot_id
        self.ask_sentences = ask_sentence.split('~')
        self.gen_dict = generative_dict
        self.nongen_dict = nongenerative_dict
        self.ngrams = defaultdict(list)
        for phrase in chain(nongenerative_dict, generative_dict):
            t = phrase.split()
            self.ngrams[len(t)].append(phrase)

        self.threshold = 95
        self.input_type = {'text'}

        self.filters = {
            'any': lambda x, _: True,
            'eq': lambda x, y: x == y,
            'not_eq': lambda x, y: x != y
        }

    def infer_from_compositional_request(self, text, input_type='text'):
        if input_type not in self.input_type:
            return None
        return self._infer_from_compositional_request(text)

    def _infer_from_compositional_request(self, text):
        return self._infer(text)

    def infer_from_single_slot(self, text, input_type='text'):
        if input_type not in self.input_type:
            return None
        return self._infer_from_single_slot(text)

    def _infer_from_single_slot(self, text):
        return self._infer(text)

    def _normal_value(self, text: str) -> str:
        return self.gen_dict.get(text, self.nongen_dict.get(text, 'Лажа какая-то'))

    def _infer(self, text: List[Dict[str, Any]]) -> Union[str, None]:
        n = len(text)
        best_score = 0
        best_candidate = None
        for window, candidates in self.ngrams.items():
            for w in range(0, n-window+1):
                query = ' '.join(x['_text'] for x in text[w:w+window])
                if query:
                    for c in candidates:
                        score = fuzz.ratio(c, query)
                        if score > best_score:
                            best_score = score
                            best_candidate = c

        if best_score >= self.threshold:
            return self._normal_value(best_candidate)

    def __repr__(self):
        return '{}(name={}, len(dict)={})'.format(self.__class__.__name__, self.id, len(self.gen_dict))

    def filter(self, value: str) -> bool:
        raise NotImplemented()

    def ask(self) -> str:
        return random.choice(self.ask_sentences)


class CurrencySlot(DictionarySlot):
    def __init__(self, slot_id: str, ask_sentence: str, generative_dict: Dict[str, str],
                 nongenerative_dict: Dict[str, str], values_order: List[str], prev_created_slots, *args):
        super().__init__(slot_id, ask_sentence, generative_dict, nongenerative_dict, values_order, prev_created_slots,
                         *args)

        self.supported_slots = ['rub', 'eur', 'usd']
        self.filters['supported_currency'] = lambda x, _: x in self.supported_slots
        self.filters['not_supported_currency'] = lambda x, _: x not in self.supported_slots


class ClassifierSlot(DictionarySlot):
    def __init__(self, slot_id: str, ask_sentence: str, generative_dict: Dict[str, str],
                 nongenerative_dict: Dict[str, str], values_order: List[str], prev_created_slots, *args):
        super().__init__(slot_id, ask_sentence, generative_dict, nongenerative_dict, values_order, prev_created_slots,
                         *args)
        self.true = values_order[0]
        self.filters.update({
            'true': lambda x, _: x == self.true,
            'false': lambda x, _: x != self.true
        })
        self.model = None

    def load_model(self, model_path):
        if not os.path.exists(model_path):
            raise Exception("Model path: '{}' doesnt exist".format(model_path))
        self.model = joblib.load(model_path)

    def train_model(self, X: List[List[Dict[str, Any]]], y, use_chars=False):
        """
        :param X: iterable with strings
        :param y: target binary labels
        :param use_chars: True if use char features
        :return: None

        """
        feat_generator = FeatureExtractor(use_chars=use_chars)
        clf = SVC()
        sticker_sent = StickSentence()
        self.model = Pipeline([("sticker_sent", sticker_sent), ('feature_extractor', feat_generator), ('svc', clf)])
        self.model.fit(X, y)

    def predict_batch(self, list_texts: List[List[Dict[str, Any]]]):
        if self.model is None:
            raise NotImplementedError("No model specified!")
        labels = self.model.predict(list_texts)
        return labels

    def _infer_from_compositional_request(self, text: List[Dict[str, Any]]):
        if self.model is None:
            raise NotImplementedError("No model specified!")
        label = bool(self.model.predict(text)[0])
        return self.true if label else None


class CompositionalSlot(DictionarySlot):
    def __init__(self, slot_id: str, ask_sentence: str, generative_dict: Dict[str, str],
                 nongenerative_dict: Dict[str, str], values_order: List[str], prev_created_slots, *args):
        super().__init__(slot_id, ask_sentence, generative_dict, nongenerative_dict, values_order, prev_created_slots,
                         *args)
        slotmap = {s.id: s for s in prev_created_slots}
        self.children = [slotmap[slot_names] for slot_names in args]
        self.input_type = set()
        for c in self.children:
            self.input_type.update(c.input_type)

    def infer_from_compositional_request(self, text, input_type='text'):
        for s in self.children:
            rv = s.infer_from_compositional_request(text, input_type)
            if rv is not None:
                return {s.id: rv, self.id: s.id}
        return None

    def infer_from_single_slot(self, text, input_type='text'):
        for s in self.children:
            rv = s.infer_from_single_slot(text, input_type)
            if rv is not None:
                return {s.id: rv, self.id: s.id}
        return None


class TomitaSlot(DictionarySlot):
    def __init__(self, slot_id: str, ask_sentence: str, generative_dict: Dict[str, str],
                 nongenerative_dict: Dict[str, str], values_order: List[str], prev_created_slots, *args):
        super().__init__(slot_id, ask_sentence, generative_dict, nongenerative_dict, values_order, prev_created_slots,
                         *args)

        config_proto = 'config.proto'
        if len(args) == 1:
            config_proto = args[0]

        config_real_path = os.path.realpath(config_proto)
        wd = os.path.dirname(config_real_path)
        assert 'TOMITA_PATH' in os.environ, 'Specify path to Tomita binary in $TOMITA_PATH'
        tomita_path = os.environ['TOMITA_PATH']
        self.tomita = Tomita(tomita_path, config_real_path, cwd=wd)

    def _infer(self, text: List[Dict[str, Any]]):
        joined_text = ' '.join(w['_text'] for w in text)
        return self.tomita.get_json(joined_text) or None


class GeoSlot(DictionarySlot):
    def __init__(self, slot_id: str, ask_sentence: str, generative_dict: Dict[str, str],
                 nongenerative_dict: Dict[str, str], values_order: List[str], prev_created_slots, *args):
        super().__init__(slot_id, ask_sentence, generative_dict, nongenerative_dict, values_order, prev_created_slots,
                         *args)
        self.input_type = {'geo'}

    def _infer(self, location: Dict[str, float]):
        return location


def read_slots_from_tsv(pipeline, filename=None):
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

        def pipe(text):
            return ' '.join([w['_text'] for w in pipeline.feed(text)])

        result_slots = []
        for row in csv_rows:
            if slot_name is None:
                slot_name, slot_class, *args = row[0].split()[0].split('->')
                info_question = row[1].strip()
                normal_names_order = []
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
                normal_name = pipe(normal_name)
                normal_names_order.append(normal_name)

                if generative_syns:
                    generative_syns = generative_syns.replace(', ', ',').replace('“', '').replace('”', '').\
                        replace('"', '').split(',')
                else:
                    generative_syns = []

                if nongenerative_syns:
                    nongenerative_syns = nongenerative_syns.replace(', ', ',').replace('“', '').replace('”', '').\
                        replace('"', '').split(',')
                else:
                    nongenerative_syns = []

                if nongenerative_syns and generative_syns:
                    assert not (set(nongenerative_syns).intersection(set(generative_syns))), [nongenerative_syns,
                                                                                              generative_syns]

                for s in nongenerative_syns:
                    nongenerative_slot_values[pipe(s)] = normal_name

                generative_slot_values[normal_name] = normal_name
                for s in generative_syns:
                    generative_slot_values[pipe(s)] = normal_name
            else:
                SlotClass = getattr(sys.modules[__name__], slot_class)
                slot = SlotClass(slot_name, info_question, generative_slot_values, nongenerative_slot_values,
                                 normal_names_order, result_slots, *args)
                result_slots.append(slot)

                slot_name = None
                generative_slot_values = {}
                nongenerative_slot_values = {}
        if slot_name:
            SlotClass = getattr(sys.modules[__name__], slot_class)
            slot = SlotClass(slot_name, info_question, generative_slot_values, nongenerative_slot_values,
                             normal_names_order, result_slots, *args)
            result_slots.append(slot)

    return result_slots


def read_slots_serialized(folder, pipe):
    """
    Read slots from tsv and load saved svm models

    :param folder: path to folder with models
    :return: array of slots

    """
    slots_array = read_slots_from_tsv(pipeline=pipe)

    for s in slots_array:
        name = os.path.join(folder, s.id + '.model')
        if isinstance(s, ClassifierSlot):
            if not os.path.exists(name):
                raise Exception("{} does not exist".format(name))
            s.load_model(name)
    return slots_array
