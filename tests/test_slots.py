import unittest
from nlu import *
from collections import defaultdict


class TestSlots(unittest.TestCase):
    def setUp(self):
        self.pipe = PreprocessorPipeline(sent_tokenize, word_tokenize, [PyMorphyPreproc(), Lower()])
        self.slots = read_slots_from_tsv(self.pipe)
        self.slots_map = {s.id: s for s in self.slots}

    def __getitem__(self, item) -> DictionarySlot:
        return self.slots_map[item]

    def test_reading_from_definitions(self):
        self.assertEqual(len(self.slots), 16)
        self.assertIn('евро', self['currency'].gen_dict)
        self.assertIn('библиотека', self['client_metro'].gen_dict)

    def test_infer_from_single_slot(self):
        self.assertEqual('савеловская', self['client_metro'].infer_from_single_slot(self.pipe.feed('рядом с метро савеловская')))

    def _test_infer_from_compositional_request(self, message_text, **slot_values):
        text = self.pipe.feed(message_text)
        target_values = defaultdict(lambda: None)
        for slot_name, slot_expected_value in slot_values.items():
            target_values[slot_name] = slot_expected_value
        for slot in self.slots:
            try:
                self.assertEqual(target_values[slot.id], slot.infer_from_compositional_request(text),
                                 msg='Slot {} expected to be "{}" but was "{}"'.format(slot.id, target_values[slot.id],
                                                                                       slot.infer_from_compositional_request(text)))
            except NotImplementedError:
                pass

    def test_infer_from_compositional_request(self):
        self._test_infer_from_compositional_request('Добрый день! Могу ли я открыть отдельный счет по 275ФЗ и что для этого нужно? ', account_type='расчетный счет')
        self._test_infer_from_compositional_request('Есть рядом с метро савеловская какое-нибудь отделение поблизости?', client_metro='савеловская')

    def test_tomita(self):
        tomita = self['client_address']
        # print(tomita.infer_from_single_slot(self.pipe.feed('ул. Маяковского, c, пятница, 22 апреля 2014 года')))
        # print(tomita.infer_from_single_slot(self.pipe.feed('улица преображенского 44')))
        # print(tomita.infer_from_single_slot(self.pipe.feed('пр. Красных Комиссаров')))
        print(tomita.infer_from_single_slot(self.pipe.feed('улица Победы 44')))


if __name__ == '__main__':
    unittest.main()