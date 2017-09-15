import os
import gzip

import csv
import json

import random
import numpy as np

from typing import List

from slots import DictionarySlot


class Sayer:

    def __init__(self, slots: List[DictionarySlot], pipe, data_dir='./nlg_data',
                 api_url='https://static-maps.yandex.ru/1.x/?l=map&pt={}'):
        self.slots = {s.id: s for s in slots}

        with open(os.path.join(data_dir, 'templates.json')) as f:
            self.templates = json.load(f)

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
                    'address': ', '.join([c for c in row[11: 14] if c]),
                    'phone': row[14],
                    'working_hours': row[15],

                    'closest_subway': self.slots['client_metro'].infer_many(pipe.feed(row[16]))
                    if row[16] and row[11] == 'г.Москва'
                    else ''
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
        if hasattr(self, method_name):
            return getattr(self, method_name)(ctx)
        return random.choice(self.templates[method_name])

    def new_acc_documents_list(self, ctx):
        docs = self.documents_data[ctx['resident']]
        template = random.choice(self.templates['new_acc_documents_list'])  # type: str
        href = docs[ctx['client_type']] if ctx['client_type'] in docs else docs['default']
        return template.format(href=href)

    def new_acc_rates_list(self, ctx):
        rates = self.rates_data[ctx['region']]
        text = 'Тарифы для выбранного региона:\n'
        if 'cities' in rates:
            text += '\n\n'.join(['<a href="{1}">{0}</a>'.format(x['title'], x['fullTableUrl']) for x in rates['cities']])
        else:
            text = 'С тарифами вы можете ознакомиться <a href="{}">по ссылке</a>'.format(rates['fullTableUrl'])
        return text

    def show_vsp(self, ctx):
        closest = []
        if ctx['method_location'] == 'client_geo':
            point = ctx['client_geo']
            point = (point['longitude'], point['latitude'])
            closest = (((self.branches_coordinates - point) ** 2).sum(axis=1) ** 0.5).argsort()[:3]
        elif ctx['method_location'] == 'client_metro':
            metro = ctx['client_metro']
            closest = [i for i in range(len(self.branches)) if metro in self.branches[i]['closest_subway']]
        text = ['Ближайшие отделения<a href="{}">:</a>']
        points = []
        n = 1
        for i in closest:
            points.append(','.join(self.branches[i]['point'] + ('pmgnm%i' % n,)))
            text.append('🏦 ' + self.branches[i]['address'])
            n += 1
        del n
        url = self.maps_api_url.format('~'.join(points))
        text[0] = text[0].format(url)
        text = '\n'.join(text)

        return text
