import json
import os
import csv
import numpy as np
import gzip


class Sayer:

    def __init__(self, slots, pipe, data_dir='./nlg_data',
                 api_url='https://static-maps.yandex.ru/1.x/?l=map&pt={}'):
        self.slots = {s.id: s for s in slots}

        with open(os.path.join(data_dir, 'new_acc_documents.json')) as f:
            self.documents_data = json.load(f)

        with gzip.open(os.path.join(data_dir, 'branches.csv.gz'), 'rt') as f:
            reader = csv.reader(f)
            next(reader)
            self.branches = []
            for row in reader:
                if not row[0]:  # Нет координат
                    continue
                self.branches.append({
                    'point': (row[0], row[1]),
                    'branch_code': row[2],
                    'branch_name': row[3],
                    'client_types': row[4],
                    'credit_in': row[5],
                    'credit_out': row[6],
                    'allow_handicapped': row[7],
                    'postcode': row[9],
                    'region': row[10],
                    'town': row[11],
                    'street': row[12],
                    'house': row[13],
                    'address': ', '.join([c for c in row[9: 14] if c]),
                    'phone': row[14],
                    'working_hours': row[15],
                    'closest_subway': row[16]
                })
        self.branches_coordinates = np.asarray([[float(c) for c in row['point']] for row in self.branches])
        self.maps_api_url = api_url

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

    def show_vsp(self, ctx):
        text = 'Не реализовано, видимо'
        if ctx['method_location'] == 'client_geo':
            point = ctx['client_geo']
            point = (point['longitude'], point['latitude'])
            closest = (((self.branches_coordinates - point) ** 2).sum(axis=1) ** 0.5).argsort()
            text = ['Ближайшие отделения:']
            points = []
            for i in closest[:3]:
                points.append(','.join(self.branches[i]['point']))
                text.append('🏦 ' + self.branches[i]['address'])
            url = self.maps_api_url.format('~'.join(points))
            text.append(url)
            text = '\n'.join(text)

        return text

    @staticmethod
    def what_now(ctx):
        return 'Мы можем вам ещё как-нибудь помочь?'

    @staticmethod
    def no_intent(ctx):
        return 'Простите, не поняла'
