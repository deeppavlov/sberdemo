from functools import lru_cache

import pymorphy2
import sys
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from typing import List, Dict, Callable
import csv
from fuzzywuzzy import process

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


class Pipeline:
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
    def __init__(self, slot_id: str, ask_sentence: str, dictionary: Dict[str, str]):
        self.id = slot_id
        self.ask_sentence = ask_sentence
        self.dict = dictionary
        self.filters = {
            'any': lambda x, _: True,
            'eq': lambda x, y: x == y,
            'not_eq': lambda x, y: x != y,
            'true': lambda x, _: bool(x),
            'false': lambda x, _: not bool(x)
        }

    def infer_from_request(self, text):
        return self._infer(text)

    def infer_from_inform(self, text):
        return self._infer(text)

    def _infer(self, text):
        """
        :param text: 
        :return: (match_norm, score)
        
        """
        norm_sent = [w['normal'] for w in text]
        matched = process.extractOne(query=' '.join(norm_sent), choices=self.dict.keys())
        if len(matched) > 0:
            return self.dict[matched[0]], matched[1]
        else:
            return ()

    def __repr__(self):
        return '{}(name={}, len(dict)={})'.format(self.__class__.__name__, self.id, len(self.dict))

    def filter(self, value: str) -> bool:
        raise NotImplemented()

    def ask(self) -> str:
        return self.ask_sentence


class ClassifierSlot(DictionarySlot):
    def __init__(self, slot_id: str, ask_sentence: str, dictionary: Dict[str, str]):
        super().__init__(slot_id, ask_sentence, dictionary)

    def infer_from_request(self, text):
        raise NotImplemented()


def read_slots_from_tsv(filename=None):
    D = '\t'
    if filename is None:
        filename = 'slots_definitions.tsv'
    with open(filename) as f:
        csv_rows = csv.reader(f, delimiter=D, quotechar='"')
        slot_name = None
        slot_class = None
        info_question = None
        slot_values = {}

        result_slots = []
        for row in csv_rows:
            if slot_name is None:
                slot_name, slot_class = row[0].split()[0].split('.')
                info_question = row[1].strip()
            elif ''.join(row):
                syns = []
                if len(row) == 1:
                    normal_name = row[0]
                elif len(row) == 2:
                    normal_name, syns = row
                    syns = syns.replace(', ', ',').replace('“', '').replace('”', '').replace('"', '').split(',')
                else:
                    raise Exception()
                slot_values[normal_name] = normal_name
                for s in syns:
                    slot_values[s] = normal_name
            else:

                SlotClass = getattr(sys.modules[__name__], slot_class)
                slot = SlotClass(slot_name, info_question, slot_values)
                result_slots.append(slot)

                slot_name = None
                slot_values = {}
        if slot_name:
            SlotClass = getattr(sys.modules[__name__], slot_class)
            slot = SlotClass(slot_name, info_question, slot_values)
            result_slots.append(slot)

    return result_slots


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
    pipe = Pipeline(sent_tokenize, word_tokenize, [PyMorphyPreproc(), Lower()], embedder=np.vstack)
    emb, text = pipe.feed('Добрый день! Могу ли я открыть отдельный счет по 275ФЗ и что для этого нужно? ')

    assert [w['_text'] for w in text] == ['добрый', 'день', '!', 'могу', 'ли', 'я', 'открыть', 'отдельный', 'счет',
                                          'по', '275фз', 'и', 'что', 'для', 'этого', 'нужно', '?']
    assert emb.shape[0] == 17, 120

    slots = read_slots_from_tsv()

    for s in slots:
        print(s.infer_from_request(text))
        print('----------')

    assert len(slots) == 9
