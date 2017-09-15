import csv
from itertools import product
from random import sample

import os

from nlu import create_pipe, read_slots_from_tsv
import re

import argparse

PARAPHRASE_DELIM = '~'

REPLICATION_FACTOR = 2

re_label_template = r'#[\w\s\d\.\/\\\,]+#\w+#'
re_label = re.compile(re_label_template)

greetings = ['Добрый день. ', 'Добрый день! ', 'Здравствуйте! ', 'Здравствуйте. ', '', '']


def generate_all_values(max_count, *slots):
    values = [list(s.gen_dict) for s in slots]
    data = list(product(*values))
    data = sample(data, min(len(data), max_count))
    for vals in data:
        yield {k: (v, k._normal_value(v)) for k, v in zip(slots, vals)}


def generate_dataset_from_templates(output_dataset_fn, generative_templates_fn):
    pipe = create_pipe()
    slots = {s.id: s for s in read_slots_from_tsv(pipe)}

    slots_global_order = sorted(slots.values(), key=lambda s: s.id)

    with open(generative_templates_fn) as fcsv:
        with open(output_dataset_fn, 'w') as f:
            print('template_id', 'intent', 'request', *[s.id for s in slots_global_order], sep='\t', file=f)
            csv_rows = csv.reader(fcsv, delimiter='\t')
            for template_id, row in enumerate(csv_rows):
                if row[0] != '1':
                    continue
                print(row[1])
                intent = row[3].strip()
                assert intent, 'Intent value can not be empty'
                for template_text in row[1].split(PARAPHRASE_DELIM):
                    slot_vals = {}
                    slots_order = []
                    for gen in re_label.findall(template_text):
                        value, slot_name = gen.strip('#').split('#')
                        assert slot_name in slots, 'Unknown slot "{}" in template "{}"'.format(slot_name, template_text)

                        text = pipe.feed(value)
                        slot = slots[slot_name]

                        slots_order.append(slot)
                        slot_vals[slot_name] = slot.infer_from_single_slot(text)

                        print(slot_name, slot.infer_from_single_slot(text), sep='=')

                    t = re_label.sub('{}', template_text)
                    for vals in generate_all_values(REPLICATION_FACTOR, *[slots[s] for s in slot_vals]):
                        msg = sample(greetings, 1)[0] + t.format(*[vals[s][0] for s in slots_order])
                        if row[2]:
                            classifiers = [x.strip() for x in row[2].split(',')]
                            for x in classifiers:
                                vals[slots[x]] = 'YES'
                        print(template_id, intent, msg, *[vals.get(s, ('',''))[1] for s in slots_global_order], sep='\t', file=f)


def main(args=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default='generated_dataset.tsv')
    parser.add_argument('--templates', default='generative_templates.tsv')

    args = parser.parse_args(args)

    assert os.path.isfile(args.templates), 'Templatesa file "{}" not found'.format(args.templates)
    if os.path.isfile(args.output):
        print('Output file "{}" is already exist. Overwriting. '.format(args.output))

    generate_dataset_from_templates(args.output, args.templates)


if __name__ == '__main__':
    main()
