import unittest
from time import time

from nlu import *
from collections import defaultdict


class TestSlots(unittest.TestCase):
    def setUp(self):
        self.pipe = PreprocessorPipeline(sent_tokenize, word_tokenize, [PyMorphyPreproc(), Lower()])
        self.slots = read_slots_from_tsv(self.pipe)  # type: List[DictionarySlot]
        self.slots_map = {s.id: s for s in self.slots}

    def __getitem__(self, item) -> DictionarySlot:
        return self.slots_map[item]

    def test_reading_from_definitions(self):
        self.assertEqual(len(self.slots), 16)
        self.assertIn('евро', self['currency'].gen_dict)
        self.assertIn('библиотека', self['client_metro'].gen_dict)

    def test_infer_from_single_slot(self):
        self.assertEqual('савеловская', self['client_metro'].infer_from_single_slot(self.pipe.feed('рядом с метро савеловская')))

    def _test_fuzzywuzzy_infer_from_compositional_request(self, message_text, **slot_values):
        text = self.pipe.feed(message_text)
        target_values = defaultdict(lambda: None)
        for slot_name, slot_expected_value in slot_values.items():
            target_values[slot_name] = slot_expected_value
        for slot in self.slots:
            if type(slot) != DictionarySlot:
                continue
            self.assertEqual(target_values[slot.id], slot.infer_from_compositional_request(text),
                             msg='Slot {} expected to be "{}" but was "{}"'.format(
                                 slot.id, slot.infer_from_compositional_request(text), target_values[slot.id]))


    def test_infer_from_compositional_request(self):
        self._test_fuzzywuzzy_infer_from_compositional_request('Добрый день! Могу ли я открыть отдельный счет по 275ФЗ и что для этого нужно? ', account_type='расчетный счет')
        self._test_fuzzywuzzy_infer_from_compositional_request('Есть рядом с метро савеловская какое-нибудь отделение поблизости?', client_metro='савеловская')

    def test_tomita(self):
        tomita = self['client_address']
        self.assertEqual('улица Победы 44', tomita.infer_from_single_slot(self.pipe.feed('улица Победы 44 проспект Сахара 33')), msg='Only first address is recognized')
        self.assertEqual('улица правды', tomita.infer_from_single_slot(self.pipe.feed('мой адрес улица правды')))
        addresses = ['улица Победы',
                     'ул Поражения д.15',
                     'ул. Правды 11',
                     'ул дружинников',
                     'сенная площадь 1',
                     'площадь революции',
                     # 'Пл джихаддистов',
                     'пр.Бешенных панд, 23',
                     'московский проспект',
                     'Ленинградский пр. 48к1',
                     'Санк-Петербургское шоссе, д. 1234',
                     'Ленинский Проезд, Дом 1234',
                     'пл морской славы 23/1',
                     'ул. маяковского 1',
                     'Первомайская ул 32/4',
                     'красная площадь',
                     'красный пр-кт',
                     'пер. Ангельский 2',
                     'Ангельский пер.',
                     'туп. Речной',
                     'тупик Речной'
                     ]
        for a in addresses:
            self.assertEqual(a.replace(',', '.').replace('. ', ' ').replace('.', ' ').strip(), tomita.infer_from_single_slot(self.pipe.feed(a)))

    def test_dictionary_slots(self):
        import pandas as pd
        table = pd.read_csv('generated_dataset.tsv', sep='\t')
        all_count = 0
        all_corrects_count = 0
        time_started = time()
        for slot in self.slots:
            if type(slot) == DictionarySlot:

                statistics = defaultdict(int)
                for row_id, row in table[['request', slot.id]].iterrows():
                    text = row['request']
                    value = row[slot.id]
                    if pd.isnull(value):
                        value = None
                    predicted = slot.infer_from_single_slot(self.pipe.feed(text))
                    correct = value == predicted
                    null_value = value is None
                    statistics[(null_value, correct)] += 1
                    all_count += 1
                    if correct:
                        all_corrects_count += 1
                    if not correct:
                        print('"{}" but was "{}" for: {}'.format(value, predicted, text))

                print('=' * 30)
                print(slot.id)
                print('    ', 'correct', 'wrong', sep='\t')
                print('nulls', statistics[(1, 1)], statistics[(1, 0)], sep='\t')
                print('filled', statistics[(0, 1)], statistics[(0, 0)], sep='\t')

        print()
        print('score is {:.2f}% for {} examples'.format(100 * all_corrects_count / all_count, all_count))
        print('\t\t{:.2f} seconds'.format(time() - time_started))

        self.assertGreater(all_corrects_count / all_count, 0.95)


if __name__ == '__main__':
    unittest.main()