import pymorphy2
from nltk.tokenize import sent_tokenize, word_tokenize
import numpy as np
from typing import List, Dict, Callable
import fasttext


# fasttext_file = '/home/marat/data/rusfasttext_on_news/model_yalen_sg_300.bin'
FASTTEXT_MODEL = '/home/marat/data/rusfasttext_on_news/ft_0.8.3_yalen_sg_300.bin'


class Preprocessor:
    def process(self, words: List[Dict]) -> List[Dict]:
        raise NotImplemented()


class Fasttext(Preprocessor):
    def __init__(self, model_path):
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
            v = np.zeros(len(self.tagmap))
            # TODO: Note index getter p[0] -- we need better disambiguation
            for tag in str(p[0].tag).replace(' ', ',').split(','):
                w[tag] = 1
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

    def feed(self, raw_input: str) -> ('embedding', List[str]):
        # TODO: is it OK to merge words from sentences?
        words = []
        for s in self.sent_tokenizer(raw_input):
            ws = [{'_text': w, '_vec': []} for w in self.word_tokenizer(s)]
            for fg in self.feature_gens:
                ws = fg.process(ws)
            words.extend(ws)

        return self.embedder([w['_vec'] for w in words]), [w['_text'] for w in words]


if __name__ == '__main__':
    pmp = PyMorphyPreproc(vectorize=False)
    assert pmp.process([{'_text': 'Разлетелся'}, {'_text': 'градиент'}]) == [{'VERB': 1, '_text': 'Разлетелся', 'indc': 1, 'past': 1, 'perf': 1, 'sing': 1, 'intr': 1 , 'masc': 1},
                                                                           {'inan': 1, '_text': 'градиент', 'masc': 1, 'sing': 1, 'nomn': 1, 'NOUN': 1}]
    lower = Lower()
    assert lower.process([{'_text': 'Разлетелся'}]) == [{'_text': 'разлетелся'}]


    # pipe = Pipeline(sent_tokenize, word_tokenize, [PyMorphyPreproc(), Lower(), Fasttext(FASTTEXT_MODEL)], embedder=np.vstack)
    pipe = Pipeline(sent_tokenize, word_tokenize, [PyMorphyPreproc(), Lower()], embedder=np.vstack)
    emb, text = pipe.feed('Добрый день! Могу ли я открыть отдельный счет по 275ФЗ и что для этого нужно? ')

    print(text)
    print(emb)



