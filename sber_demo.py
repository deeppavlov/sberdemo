from nlu import *

SLOT_ACCOUNT_RESERVATION = 'account_reservation'
SLOT_ACCOUNT_CURRENCY = 'account_currency'
SLOT_RESIDENT = 'resident'
SLOT_WANT_SEE_DOCS = 'see_docs'
SLOT_FORMA_SOBST = 'forma_sobstv'


def some_starts_with(words_list, *prefixi_to_check):
    for w in words_list:
        if w.startswith(prefixi_to_check):
            return True
    return False


class RuleBasedSberdemoNLU:
    def forward(self, embedding, text):
        res = {'slots': {}}

        normals = {w['normal'] for w in text}
        intent = None
        if {'да', 'давай', 'давайте', 'конечно', 'хорошо', 'ладно', 'ок', 'ok'}.intersection(set(w['_text'] for w in text)):
            res['confirmation'] = True
        if {'нет'}.intersection(set(w['_text'] for w in text)):
            res['confirmation'] = False

        if {'открыть', 'счет'}.issubset(normals):
            intent = 'OpenAccount'

        slot_values = {SLOT_ACCOUNT_CURRENCY: None,
                       'slot_address': None,
                       'see_docs': None}

        words = [w['_text'] for w in text]
        if some_starts_with(words, 'рубл', 'rub'):
            slot_values[SLOT_ACCOUNT_CURRENCY] = 'RUB'
        elif some_starts_with(words, 'евро', 'EUR'):
            slot_values[SLOT_ACCOUNT_CURRENCY] = 'EUR'
        elif some_starts_with(words, 'доллар', 'долар', 'бакс', 'usd', '$', 'dollar'):
            slot_values[SLOT_ACCOUNT_CURRENCY] = 'USD'

        if some_starts_with(words, 'документ'):
            slot_values[SLOT_WANT_SEE_DOCS] = True

        if some_starts_with(words, 'ooo', 'ооо'):
            slot_values[SLOT_FORMA_SOBST] = 'ооо'
        if some_starts_with(words, 'зао'):
            slot_values[SLOT_FORMA_SOBST] = 'зао'

        if intent is not None:
            res['intent'] = intent
        for k, v in slot_values.items():
            if v:
                res['slots'][k] = v

        return res


class RuleBasedSberdemoPolicy:
    def __init__(self):
        self.slots = {}
        self.expect_confirmation = None

    def forward(self, client_nlu):
        if self.expect_confirmation is not None:
            if 'confirmation' in client_nlu:
                self.slots[self.expect_confirmation] = client_nlu['confirmation']
                self.expect_confirmation = None

        if 'intent' in client_nlu:
            self.intent = client_nlu['intent']

        if self.intent is None:
            return 'Извините, я не понял! Что хотите сделтать?'

        if self.intent == 'OpenAccount':
            self._obtain_slots(client_nlu['slots'], SLOT_ACCOUNT_CURRENCY, SLOT_WANT_SEE_DOCS, SLOT_RESIDENT, SLOT_FORMA_SOBST)

            if self.slots.get(SLOT_ACCOUNT_CURRENCY, None) is None:
                return 'В какой валюте хотите открыть счёт?'
            elif self.slots[SLOT_ACCOUNT_CURRENCY] != 'RUB':
                return 'Открыть счёт в иностранной валюте можно только в офисе Сбербанка. Найти для вас ближайшее отделение банка, где вы сможете открыть валютный счёт?'
            else:  # client wants ruble account!
                if self.slots.get(SLOT_WANT_SEE_DOCS, None) is None:
                    self.expect_confirmation = SLOT_WANT_SEE_DOCS
                    return 'Вы хотели бы ознакомиться с необходимым перечнем документов и тарифами на обслуживание?'

                if self.slots[SLOT_WANT_SEE_DOCS] and not self.slots.get('docs_were_shown', None):
                    if self.slots.get(SLOT_RESIDENT, None) is None:
                        self.expect_confirmation = SLOT_RESIDENT
                        return 'Вы являетесь резидентом РФ?'

                    if self.slots.get(SLOT_FORMA_SOBST, None) is None:
                        self.expect_confirmation = SLOT_FORMA_SOBST
                        return 'Уточните Вашу форму собственности?'

                    self.slots['docs_were_shown'] = True
                    self.expect_confirmation = SLOT_ACCOUNT_RESERVATION
                    return '''ПоказатьПереченьДокументов({}={},{}={})
Предлагаю Вам прямо сейчас зарезервировать счёт в режиме диалога. <одно из преимуществ>
Для этого нужно предоставить ОГРН (ОГРИН), ИНН, КПП (ЮЛ), данные документа, удостоверяющего личность, дату рождения руководителя и так же имеется на данный момент доступ к электронной почте?
Вы хотели бы сейчас произвести онлайн резервирование счёта?'''.format(SLOT_RESIDENT, self.slots[SLOT_RESIDENT],
                                                           SLOT_FORMA_SOBST, self.slots[SLOT_FORMA_SOBST])

                if self.slots.get(SLOT_ACCOUNT_RESERVATION, None) is None:
                    self.expect_confirmation = SLOT_ACCOUNT_RESERVATION
                    return '''Предлагаю Вам прямо сейчас зарезервировать счёт в режиме диалога. <одно из преимуществ>
Для этого нужно предоставить ОГРН (ОГРИН), ИНН, КПП (ЮЛ), данные документа, удостоверяющего личность, дату рождения руководителя и так же имеется на данный момент доступ к электронной почте?
Вы хотели бы сейчас произвести онлайн резервирование счёта?'''

                if self.slots[SLOT_ACCOUNT_RESERVATION]:
                    return 'НачатьРезервированиеCчёта()\nВсего хорошего, до свидания!'

        return 'Всего доброго!'

    def _obtain_slots(self, new_slots, *slots_to_obtain):
        for slot_name in slots_to_obtain:
            if slot_name in new_slots:
                self.slots[slot_name] = new_slots[slot_name]


class Dialog:
    def __init__(self, preproc_pipeline, nlu_model, policy_model):
        self.pipeline = preproc_pipeline
        self.nlu_model = nlu_model
        self.policy_model = policy_model

    def forward(self, client_utterance):
        print('>>>', client_utterance)
        emb, text = self.pipeline.feed(client_utterance)
        nlu_result = self.nlu_model.forward(emb, text)
        print(nlu_result)
        response = self.policy_model.forward(nlu_result)
        print('<<<', response)
        return response


def _assert(expected, actual):
    assert expected == actual, 'Expected "{}", but was "{}"'.format(expected, actual)

if __name__ == '__main__':
    pipe = Pipeline(sent_tokenize, word_tokenize, [PyMorphyPreproc(), Lower()], embedder=np.vstack)
    dialog = Dialog(pipe, RuleBasedSberdemoNLU(), RuleBasedSberdemoPolicy())

    resp = dialog.forward('Добрый день! Могу ли я открыть отдельный счет по 275ФЗ и что для этого нужно?')
    _assert('В какой валюте хотите открыть счёт?', resp)
    resp = dialog.forward('Российский рубль!')
    _assert('Вы хотели бы ознакомиться с необходимым перечнем документов и тарифами на обслуживание?', resp)
    resp = dialog.forward('Конечно!')
    _assert('Вы являетесь резидентом РФ?', resp)
    resp = dialog.forward('да')
    _assert('Уточните Вашу форму собственности?', resp)
    resp = dialog.forward('ООО')
    _assert('''ПоказатьПереченьДокументов(resident=True,forma_sobstv=ооо)
Предлагаю Вам прямо сейчас зарезервировать счёт в режиме диалога. <одно из преимуществ>
Для этого нужно предоставить ОГРН (ОГРИН), ИНН, КПП (ЮЛ), данные документа, удостоверяющего личность, дату рождения руководителя и так же имеется на данный момент доступ к электронной почте?
Вы хотели бы сейчас произвести онлайн резервирование счёта?''', resp)
    resp = dialog.forward('да')
    _assert('НачатьРезервированиеCчёта()\nВсего хорошего, до свидания!', resp)
