from functools import lru_cache

import pymorphy2
from gensim.models.wrappers import FastText
from typing import List, Dict, Callable

from intent_classifier import IntentClassifier
from slots import read_slots_from_tsv, DictionarySlot
from nltk.tokenize import sent_tokenize, word_tokenize
from svm_classifier_utlilities import *
from tomita.name_parser import NameParser


class Preprocessor:
    def process(self, words: List[Dict]) -> List[Dict]:
        raise NotImplemented()


class FastTextPreproc(Preprocessor):
    def __init__(self, model_path, normalized=True):
        self.k = 'normal' if normalized else '_text'
        self.model = FastText.load(model_path)
        self.zero = np.zeros(self.model.vector_size)

    def process(self, words: List[Dict]):
        for w in words:
            try:
                w['_vec'].append(self.model[w[self.k]])
            except KeyError:
                w['_vec'].append(self.zero)
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


class Replacer(Preprocessor):
    def __init__(self, *replacement_from_to_pairs):
        self.pairs = replacement_from_to_pairs

    def process(self, words: List[Dict]):
        res = []
        for w in words:
            for old, new in self.pairs:
                w['_text'] = w['_text'].replace(old, new)
            res.append(w)
        return res


class PreprocessorPipeline:
    def __init__(self,
                 sent_tokenizer: Callable[[str], List[str]],
                 word_tokenizer: Callable[[str], List[str]],
                 feature_gens: List[Preprocessor]):
        self.sent_tokenizer = sent_tokenizer
        self.word_tokenizer = word_tokenizer
        self.feature_gens = feature_gens

    @lru_cache()
    def feed(self, raw_input: str) -> List[Dict]:
        # TODO: is it OK to merge words from sentences?
        words = []
        for s in self.sent_tokenizer(raw_input):
            ws = [{'_text': w, '_orig': w, '_vec': []} for w in self.word_tokenizer(s)]
            for fg in self.feature_gens:
                ws = fg.process(ws)
            if ws:
                words.extend(ws)

        return words


class StatisticalNLUModel:
    def __init__(self, slots: List[DictionarySlot], intent_classifier: IntentClassifier, name_parser: NameParser):
        self.slots = {s.id: s for s in slots}  # type: Dict[str, DictionarySlot]
        self.intent_classifier = intent_classifier
        self.expect = None

        self.expect_name = True
        self.name_parser = name_parser

    def forward(self, message, message_type='text'):
        res = {
            'slots': {}
        }

        if self.expect_name and message_type == 'text':
            name = self.name_parser.parse(message)
            if isinstance(name, list):
                name = name[0]
            res['name'] = name
            self.expect_name = False

        if self.expect is None:
            if message_type == 'text':
                res['intent'] = self.intent_classifier.predict_single(message)
            else:
                res['intent'] = 'no_intent'

            for slot in self.slots.values():
                val = slot.infer_from_compositional_request(message, message_type)
                if isinstance(val, dict):
                    res['slots'].update(val)
                elif val is not None:
                    res['slots'][slot.id] = val
        else:

            for slot in self.slots.values():
                val = slot.infer_from_compositional_request(message, message_type)
                if isinstance(val, dict):
                    res['slots'].update(val)
                elif val is not None:
                    res['slots'][slot.id] = val

            slot = self.slots[self.expect]
            val = slot.infer_from_single_slot(message, message_type)
            if isinstance(val, dict):
                res['slots'].update(val)
            elif val is not None:
                res['slots'][self.expect] = val

        return res

    def set_expectation(self, expect):
        self.expect = expect


def create_pipe(fasttext_model_path=None):
    preprocessors = [Replacer(('ё', 'е')), PyMorphyPreproc(), Lower()]
    if fasttext_model_path:
        preprocessors.append(FastTextPreproc(model_path=fasttext_model_path))

    return PreprocessorPipeline(sent_tokenize, word_tokenize, preprocessors)


if __name__ == '__main__':
    pass


