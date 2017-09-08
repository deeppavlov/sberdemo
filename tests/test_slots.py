import unittest
from nlu import *


class TestSlots(unittest.TestCase):
    def setUp(self):
        self.pipe = PreprocessorPipeline(sent_tokenize, word_tokenize, [PyMorphyPreproc(), Lower()])
        self.slots = read_slots_from_tsv(self.pipe)
        self.slots_map = {s.id: s for s in self.slots}

    def __getitem__(self, item):
        return self.slots_map[item]

    def test_reading_from_definitions(self):
        self.assertEqual(len(self.slots), 14)
        self.assertIn('евро', self['currency'].gen_dict)
        self.assertIn('библиотека', self['client_metro'].gen_dict)

    def test_infer_from_single_slot(self):
        self.assertEqual('савеловская', self['client_metro'].infer_from_single_slot(self.pipe.feed('рядом с метро савеловская')))

    def test_infer_from_compositional_request(self):
        text = self.pipe.feed('Добрый день! Могу ли я открыть отдельный счет по 275ФЗ и что для этого нужно? ')
        target_values = defaultdict(lambda: None)
        target_values['account_type'] = 'расчетный счет'
        for slot in self.slots:
            try:
                self.assertEqual(target_values[slot.id], slot.infer_from_compositional_request(text))
            except NotImplementedError:
                pass

        raw_text = 'Есть рядом с метро савеловская какое-нибудь отделение поблизости?'
        text = self.pipe.feed(raw_text)
        target_values = defaultdict(lambda: None)
        target_values['client_metro'] = 'савеловская'
        for slot in self.slots:
            try:
                self.assertEqual(target_values[slot.id], slot.infer_from_compositional_request(text), raw_text)
            except NotImplementedError:
                pass


if __name__ == '__main__':
    unittest.main()