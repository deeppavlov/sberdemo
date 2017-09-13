import json
import os


class Sayer:

    def __init__(self, slots, pipe, data_dir='./nlg_data'):
        self.slots = {s.id: s for s in slots}

        with open(os.path.join(data_dir, 'new_acc_documents.json')) as f:
            self.documents_data = json.load(f)

        self.rates_data = {}
        with open(os.path.join(data_dir, 'rates_urls.json')) as f:
            rates_data = json.load(f)
        for rate in rates_data:
            norm = self.slots['region'].infer_from_single_slot(pipe.feed(rate['title']))
            if norm:
                self.rates_data[norm] = rate

    def say(self, method_name, ctx):
        return getattr(self, method_name)(ctx)

    @staticmethod
    def cant_reserve(ctx):
        return 'Нельзя резервировать счёт не в рублях'

    def new_acc_documents_list(self, ctx):
        docs = self.documents_data[ctx['resident']]
        text = 'С необходимыми документами вы можете ознакомиться по ссылке: '
        text += docs[ctx['client_type']] if ctx['client_type'] in docs else docs['default']
        return text

    def new_acc_rates_list(self, ctx):
        rates = self.rates_data[ctx['region']]
        text = 'Тарифы для выбранного региона:\n'
        if 'cities' in rates:
            text += '\n\n'.join(['{}: {}'.format(x['title'], x['fullTableUrl']) for x in rates['cities']])
        else:
            text += '{}'.format(rates['fullTableUrl'])
        return text

    @staticmethod
    def not_supported(ctx):
        return 'Такая валюта не поддерживается. Можно открыть в рублях, долларах и евро'

    @staticmethod
    def send_to_bank(ctx):
        return 'Для открытия счёта обратись в отеление Сбербанка'

    @staticmethod
    def reserve_new_acc_online(ctx):
        return 'Зарезервировать счёт вы можете по ссылке: ' \
               'https://www.sberbank.ru/ru/s_m_business/bankingservice/rko/service23'

    @staticmethod
    def weird_route(ctx):
        return 'You were not supposed to see this'

    @staticmethod
    def show_vsp(ctx):
        return '`точки на карте с отеделениями`'

    @staticmethod
    def what_now(ctx):
        return 'Мы можем вам ещё как-нибудь помочь?'

    @staticmethod
    def no_intent(ctx):
        return 'Простите, не поняла'
