from functools import lru_cache

import pymorphy2
from typing import List, Dict, Callable

from intent_classifier import IntentClassifier
from slots import read_slots_from_tsv, DictionarySlot
from nltk.tokenize import sent_tokenize, word_tokenize
from svm_classifier_utlilities import *

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
    def feed(self, raw_input: str) -> List[str]:
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
    def __init__(self, slots: List[DictionarySlot], intent_classifier: IntentClassifier):
        self.slots = {s.id: s for s in slots}  # type: Dict[str, DictionarySlot]
        self.intent_classifier = intent_classifier
        self.expect = None

    def forward(self, message, message_type='text'):
        res = {
            'slots': {}
        }

        if self.expect is None:
            res['intent'] = self.intent_classifier.predict_single(message)

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


def create_pipe():
    return PreprocessorPipeline(sent_tokenize, word_tokenize, [Replacer(('ё', 'е')),
                                                               PyMorphyPreproc(),
                                                               Lower()])


if __name__ == '__main__':
    pass


