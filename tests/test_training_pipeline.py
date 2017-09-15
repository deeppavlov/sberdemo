import tempfile

import os
import subprocess
from subprocess import PIPE

import shutil

import nlu
import slots


def run(*args):
    print('Run: ', *args)
    return subprocess.run(args, stderr=PIPE, stdout=PIPE, check=True)

assert 'TOMITA_PATH' in os.environ, 'Please specify path to Tomita Parser in $TOMITA_PATH'

dir = tempfile.TemporaryDirectory(prefix='sberdemotest_').name
os.mkdir(dir)

dataset = os.path.join(dir, 'dataset.tsv')
templates = 'generative_templates.tsv'
slot_definitions = 'slots_definitions.tsv'

assert os.path.isfile('generate_from_templates.py'), '"generate_from_templates.py" not found'

run('python3', 'generate_from_templates.py', '--output', dataset, '--templates', templates)

with open(dataset) as f:
    for line in f:
        assert '#' not in line.strip(), 'Found "#" in line "{}"'.format(line)
print('Dataset successfully created in "{}"'.format(dataset))


run('python3', 'train_svm.py', '--folder', dir, '--data', dataset, '--slot_path', slot_definitions, '--slot_train')
print('slots trained')
run('python3', 'train_svm.py', '--folder', dir, '--data', dataset, '--slot_path', slot_definitions, '--intent_train')
print('intent recognizer trained')

pipe = nlu.create_pipe()
slots.read_slots_serialized(dir, pipe)
print('Slots deserialized')

print()

print('The test has just passed!')

print()

print('cleaning up...')
shutil.rmtree(dir)
print('done')


