from collections import defaultdict
from functools import lru_cache
from itertools import chain

import pymorphy2
import sys
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from typing import List, Dict, Callable, Any, Union
import csv
from fuzzywuzzy import fuzz
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from slots_classifier_utlilities import *
import os
from sklearn.externals import joblib

# fasttext_file = '/home/marat/data/rusfasttext_on_news/model_yalen_sg_300.bin'
FASTTEXT_MODEL = '/home/marat/data/rusfasttext_on_news/ft_0.8.3_yalen_sg_300.bin'


class Preprocessor:
    def process(self, words: List[Dict]) -> List[Dict]:
        raise NotImplemented()


class Fasttext(Preprocessor):
    def __init__(self, model_path):
        import fasttext
        self.model = fasttext.load_model(model_path)

    def process(self, words: List[Dict]):
        for w in words:
            w['_vec'].append(self.model[w['_text']])
        return words


class PyMorphyPreproc(Preprocessor):
    def __init__(self, vectorize=True):
        self.vectorize = vectorize
        self.morph = pymorphy2.MorphAnalyzer()
        tags = sorted(self.morph.dictionary.Tag.KNOWN_GRAMMEMES)
        self.tagmap = dict(zip(tags, range(len(tags))))

    def process(self, words):
        res = []
        for w in words:
            p = self.morph.parse(w['_text'])
            w['normal'] = p[0].normal_form.replace('ё', 'е')
            v = np.zeros(len(self.tagmap))
            # TODO: Note index getter p[0] -- we need better disambiguation
            for tag in str(p[0].tag).replace(' ', ',').split(','):
                w['t_' + tag] = 1
                v[self.tagmap[tag]] = 1
            if self.vectorize:
                w['_vec'].append(v)
            res.append(w)
        return res


class Lower(Preprocessor):
    def process(self, words):
        res = []
        for w in words:
            w['_text'] = w['_text'].lower()
            res.append(w)
        return res


class Pipeline_nlp:
    def __init__(self,
                 sent_tokenizer: Callable[[str], List[str]],
                 word_tokenizer: Callable[[str], List[str]],
                 feature_gens: List[Preprocessor],
                 embedder: Callable):
        self.sent_tokenizer = sent_tokenizer
        self.word_tokenizer = word_tokenizer
        self.feature_gens = feature_gens
        self.embedder = embedder

    @lru_cache()
    def feed(self, raw_input: str) -> ('embedding', List[str]):
        # TODO: is it OK to merge words from sentences?
        words = []
        for s in self.sent_tokenizer(raw_input):
            ws = [{'_text': w, '_vec': []} for w in self.word_tokenizer(s)]
            for fg in self.feature_gens:
                ws = fg.process(ws)
            if ws:
                words.extend(ws)

        return self.embedder([w['_vec'] for w in words]), words


class DictionarySlot:
    def __init__(self, slot_id: str, ask_sentence: str, generative_dict: Dict[str, str], nongenerative_dict: Dict[str, str]):
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
    def __init__(self, slot_id: str, ask_sentence: str, generative_dict: Dict[str, str], nongenerative_dict: Dict[str, str]):
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
    def __init__(self, slot_id: str, ask_sentence: str, generative_dict: Dict[str, str], nongenerative_dict: Dict[str, str]):
        super().__init__(slot_id, ask_sentence, generative_dict, nongenerative_dict)


class TomitaSlot(DictionarySlot):
    def __init__(self, slot_id: str, ask_sentence: str, generative_dict: Dict[str, str], nongenerative_dict: Dict[str, str]):
        super().__init__(slot_id, ask_sentence, generative_dict, nongenerative_dict)


class GeoSlot(DictionarySlot):
    def __init__(self, slot_id: str, ask_sentence: str, generative_dict: Dict[str, str], nongenerative_dict: Dict[str, str]):
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
                    generative_syns = generative_syns.replace(', ', ',').replace('“', '').replace('”', '').replace('"', '').split(',')
                else:
                    generative_syns = []

                if nongenerative_syns:
                    nongenerative_syns = nongenerative_syns.replace(', ', ',').replace('“', '').replace('”', '').replace('"', '').split(',')
                else:
                    nongenerative_syns = []

                if nongenerative_syns and generative_syns:
                    assert not (set(nongenerative_syns).intersection(set(generative_syns))), [nongenerative_syns, generative_syns]

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


if __name__ == '__main__':
    pmp = PyMorphyPreproc(vectorize=False)
    assert pmp.process([{'_text': 'Разлетелся'}, {'_text': 'градиент'}]) == [{'t_intr': 1, 't_VERB': 1, 't_indc': 1,
                                                                              'normal': 'разлететься', 't_past': 1,
                                                                              't_sing': 1, '_text': 'Разлетелся',
                                                                              't_perf': 1, 't_masc': 1},
                                                                             {'t_sing': 1, 't_NOUN': 1,
                                                                              'normal': 'градиент', '_text': 'градиент',
                                                                              't_nomn': 1, 't_inan': 1, 't_masc': 1}]

    lower = Lower()
    assert lower.process([{'_text': 'Разлетелся'}]) == [{'_text': 'разлетелся'}]

    # pipe = Pipeline(sent_tokenize, word_tokenize, [PyMorphyPreproc(), Lower(), Fasttext(FASTTEXT_MODEL)], embedder=np.vstack)
    pipe = Pipeline_nlp(sent_tokenize, word_tokenize, [PyMorphyPreproc(), Lower()], embedder=np.vstack)
    test_input_str = 'Добрый день! Могу ли я открыть отдельный счет по 275ФЗ и что для этого нужно? '
    emb, text = pipe.feed(test_input_str)

    assert [w['_text'] for w in text] == ['добрый', 'день', '!', 'могу', 'ли', 'я', 'открыть', 'отдельный', 'счет',
                                          'по', '275фз', 'и', 'что', 'для', 'этого', 'нужно', '?']
    assert emb.shape[0] == 17, 120

    slots = read_slots_from_tsv()
    assert len(slots) == 14, len(slots)

    slotmap = {s.id:s for s in slots}

    assert 'евро' in slotmap['currency'].gen_dict
    assert 'библиотека' in slotmap['client_metro'].gen_dict

    # slotmap['client_metro'].infer_from_composional_request(pipe.feed('Есть рядом с метро савеловская какое-нибудь отделение поблизости?')[1])
    slotmap['client_metro'].infer_from_single_slot(pipe.feed('рядом с метро савеловская')[1])

    print('='*30)
    print('compositional infer for "{}"'.format(test_input_str))
    for s in slots:
        try:
            print(s.infer_from_compositional_request(text))
            print('----------')
        except NotImplementedError:
            print('Infer not implemented for slot "{}"'.format(s.id))


