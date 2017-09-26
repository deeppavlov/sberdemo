import os
import gzip

import csv
import json

import random
import numpy as np

from typing import List

from slots import DictionarySlot

import urllib.request
from urllib.parse import urlencode

import random


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
        params = {
            'name_front': ''
        }
        params.update(ctx)
        if params.get('client_name') and random.random() < 0.3:
            params['name_front'] = '{}, '.format(params['client_name']['formal'])
        res = random.choice(self.templates[method_name]).format(**params)
        return res[0].upper() + res[1:]

    def new_acc_documents_list(self, ctx):
        docs = self.documents_data[ctx['resident']]
        template = random.choice(self.templates['new_acc_documents_list'])  # type: str
        href = docs[ctx['client_type']] if ctx['client_type'] in docs else docs['default']
        return template.format(href=href)

    def new_acc_rates_list(self, ctx):
        templates = self.templates['new_acc_rates_list']
        if ctx['region'] not in self.rates_data:
            return templates['not_found']
        rates = self.rates_data[ctx['region']]
        if 'cities' in rates:
            templates = templates['multiple']
            text = templates['start']
            text += '\n'.join([templates['city'].format(city=x['title'], href=x['fullTableUrl']) for x in rates['cities']])
        else:
            text = templates['single'].format(href=rates['fullTableUrl'])
        return text

    def show_vsp(self, ctx):
        templates = self.templates['show_vsp']
        closest = []
        if ctx['method_location'] == 'client_address':
            params = {
                'format': 'json',
                'bbox': '37.32624,55.491126~37.967682,55.957565',
                'rspn': 1,
                'geocode': '+'.join(ctx['client_address'].split(' '))
            }
            url = 'https://geocode-maps.yandex.ru/1.x/?' + urlencode(params)
            response = json.loads(urllib.request.urlopen(url).read().decode('UTF8'))
            response = response['response']['GeoObjectCollection']['featureMember']
            if response:
                point = [float(c) for c in response[0]['GeoObject']['Point']['pos'].split(' ')]
                closest = (((self.branches_coordinates - point) ** 2).sum(axis=1) ** 0.5).argsort()[:3]
        elif ctx['method_location'] == 'client_geo':
            point = ctx['client_geo']
            point = (point['longitude'], point['latitude'])
            closest = (((self.branches_coordinates - point) ** 2).sum(axis=1) ** 0.5).argsort()[:3]
        elif ctx['method_location'] == 'client_metro':
            metro = ctx['client_metro']
            closest = [i for i in range(len(self.branches)) if metro in self.branches[i]['closest_subway']]

        if len(closest) == 0:
            return templates['not_found']

        text = [templates['start']]
        points = []
        n = 1
        for i in closest:
            points.append(','.join(self.branches[i]['point'] + ('pmgnm%i' % n,)))
            branch = [self.branches[i]['address']]
            if ctx.get('show_schedule') == self.slots['show_schedule'].true:
                working_hours = ['🕒 ' + _ for _ in self.branches[i]['working_hours'].split(', ')]
                branch.append('\n'.join(working_hours))
            if ctx.get('show_phone') == self.slots['show_phone'].true:
                branch.append('📞 ' + self.branches[i]['phone'])
            text.append(templates['branch'].format(branch='\n'.join(branch)))
            n += 1
        del n
        url = self.maps_api_url.format('~'.join(points))
        text[0] = text[0].format(href=url)
        text = '\n\n'.join(text)

        return text
