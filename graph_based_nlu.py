from slots import *
from nlu import *


def some_starts_with(words_list, *prefixi_to_check):
    for w in words_list:
        if w.startswith(prefixi_to_check):
            return True
    return False


class GraphBasedSberdemoNLU:

    def __init__(self):
        self.expect = None

    def set_expectation(self, expect):
        self.expect = expect

    def forward(self, embedding, text: List[Dict]):
        res = {'slots': {}}

        normals = {w['normal'] for w in text}
        intent = None

        slot_values = {NEW_ACC_CURRENCY: None,
                       'slot_address': None,
                       'see_docs': None}

        if {'да', 'давай', 'давайте', 'конечно', 'хорошо', 'ладно', 'ок', 'ok'}.intersection(set(w['_text'] for w in text)):
            res['confirmation'] = True
        if {'нет'}.intersection(set(w['_text'] for w in text)):
            res['confirmation'] = False

        if self.expect and 'confirmation' in res:
            res['slots'][self.expect] = res['confirmation']
            del res['confirmation']

        if {'открыть', 'счет'}.issubset(normals):
            intent = 'new_acc_intent'

        words = [w['_text'] for w in text]
        if some_starts_with(words, 'рубл', 'rub'):
            slot_values[NEW_ACC_CURRENCY] = 'RUB'
        elif some_starts_with(words, 'евро', 'EUR'):
            slot_values[NEW_ACC_CURRENCY] = 'EUR'
        elif some_starts_with(words, 'доллар', 'долар', 'бакс', 'usd', '$', 'dollar'):
            slot_values[NEW_ACC_CURRENCY] = 'USD'

        if some_starts_with(words, 'документ'):
            slot_values[NEW_ACC_SHOW_DOCS] = True

        if some_starts_with(words, 'ooo', 'ооо'):
            slot_values[NEW_ACC_OWNERSHIP_FORM] = 'ооо'
        if some_starts_with(words, 'зао'):
            slot_values[NEW_ACC_OWNERSHIP_FORM] = 'зао'

        if intent is not None:
            res['intent'] = intent
        for k, v in slot_values.items():
            if v:
                res['slots'][k] = v

        return res
