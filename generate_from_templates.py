import csv
from itertools import product
from random import choice, sample

from nltk import sent_tokenize, word_tokenize

from nlu import read_slots_from_tsv, Pipeline, PyMorphyPreproc, Lower
import re

REPLICATION_FACTOR = 10

re_label_template = r'#[\w\s\d\.]+#\w+#'
re_label = re.compile(re_label_template)

greetings = ['Добрый день. ', 'Добрый день! ', 'Здравствуйте! ', 'Здравствуйте. ', '', '']


def generate_all_values(max_count, *slots):
    values = [list(s.dict) for s in slots]
    data = list(product(*values))
    data = sample(data, min(len(data), max_count))
    for vals in data:
        yield {k: v for k, v in zip(slots, vals)}

if __name__ == '__main__':
    slots = {s.id: s for s in read_slots_from_tsv()}
    slots_global_order = sorted(slots.values(), key=lambda s: s.id)

    pipe = Pipeline(sent_tokenize, word_tokenize, [PyMorphyPreproc(), Lower()], embedder=lambda _: None)

    templates = []

    with open('generative_templates.tsv') as fcsv:
        with open('generated_dataset.tsv', 'w') as f:
            print('template_id', 'request', *[s.id for s in slots_global_order], sep='\t', file=f)
            csv_rows = csv.reader(fcsv, delimiter='\t')
            for template_id, row in enumerate(csv_rows):
                if row[0] != '1':
                    continue
                print(row[1])
                slot_vals = {}
                slots_order = []
                for gen in re_label.findall(row[1]):
                    value, slot_name = gen.strip('#').split('#')
                    assert slot_name in slots, 'Unknown slot "{}" in templates'.format(slot_name)

                    _, text = pipe.feed(value)
                    slot = slots[slot_name]

                    slots_order.append(slot)
                    slot_vals[slot_name] = slot.infer_from_single_slot(text)

                    print(slot_name, slot.infer_from_single_slot(text), sep='=')

                t = re_label.sub('{}', row[1])
                for vals in generate_all_values(REPLICATION_FACTOR, *[slots[s] for s in slot_vals]):
                    msg = sample(greetings, 1)[0] + t.format(*[vals[s] for s in slots_order])
                    if row[2]:
                        classifiers = [x.strip() for x in row[2].split(',')]
                        for x in classifiers:
                            vals[slots[x]] = 'YES'
                    print(template_id, msg, *[vals.get(s, '') for s in slots_global_order], sep='\t', file=f)
