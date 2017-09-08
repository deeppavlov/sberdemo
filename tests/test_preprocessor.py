import unittest
from nlu import *

class PreprocessorPipelineTest(unittest.TestCase):
    def test_lower(self):
        lower = Lower()
        self.assertEqual(lower.process([{'_text': 'Разлетелся'}]), [{'_text': 'разлетелся'}])

    def test_PyMorphyPreproc(self):
        pmp = PyMorphyPreproc(vectorize=False)
        self.assertEqual(pmp.process([{'_text': 'Разлетелся'}, {'_text': 'градиент'}]), [{'t_intr': 1, 't_VERB': 1, 't_indc': 1,
                                                                                  'normal': 'разлететься', 't_past': 1,
                                                                                  't_sing': 1, '_text': 'Разлетелся',
                                                                                  't_perf': 1, 't_masc': 1},
                                                                                 {'t_sing': 1, 't_NOUN': 1,
                                                                                  'normal': 'градиент',
                                                                                  '_text': 'градиент',
                                                                                  't_nomn': 1, 't_inan': 1,
                                                                                  't_masc': 1}])

    def test_tokenizers(self):
        pipe = PreprocessorPipeline(sent_tokenize, word_tokenize, [PyMorphyPreproc(), Lower()])
        test_input_str = 'Добрый день! Могу ли я открыть отдельный счет по 275ФЗ и что для этого нужно? '
        text = pipe.feed(test_input_str)

        self.assertEqual([w['_text'] for w in text], ['добрый', 'день', '!', 'могу', 'ли', 'я', 'открыть', 'отдельный',
                                                      'счет', 'по', '275фз', 'и', 'что', 'для', 'этого', 'нужно', '?'])


if __name__ == '__main__':
    unittest.main()
