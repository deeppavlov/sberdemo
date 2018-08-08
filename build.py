import urllib.request
from time import time

# from train_joint_classifier import main as train_joint_classifier
from generate_from_templates import main as generate_from_templates
from train_svm import main as train_svm
from nlu import create_pipe
from extend_spell_checker_dict import main as extend_spell_checker

import os
import shutil

import gzip


def main():
    time_start = time()
    root = os.path.dirname(os.path.realpath(__file__))
    models_dir = 'models_nlu'
    directory = os.path.join(root, models_dir)

    templates = 'generative_templates.tsv'
    slot_definitions = 'slots_definitions.tsv'
    no_intent_dataset = 'sberdemo_no_intent.tsv.gz'

    dataset = os.path.join(root, 'generated_dataset.tsv')

    shutil.rmtree(directory, ignore_errors=True)
    try:
        os.remove(dataset)
    except OSError:
        pass

    os.mkdir(directory)

    no_intent_preprocessed = 'no_intent_corpus.txt'
    if os.path.isfile(no_intent_preprocessed):
        print('Using old', no_intent_preprocessed)
    else:
        if not os.path.isfile(no_intent_dataset):
            url = 'ftp://share.ipavlov.mipt.ru/datasets/' + os.path.basename(no_intent_dataset)
            try:
                urllib.request.urlretrieve(url, no_intent_dataset)
            except:
                pass

        pipe = create_pipe(fasttext_model_path=None)
        with open(no_intent_preprocessed, 'w') as out:
            with gzip.open(no_intent_dataset, 'rt', encoding='UTF8') as f:
                for line in f:
                    words = pipe.feed(line.strip())
                    print(' '.join(w['_text'] for w in words), file=out)
        print('{} has been preprocessed'.format(no_intent_dataset))

    print()

    extend_spell_checker()

    args = ['--output', dataset, '--templates', templates]
    generate_from_templates(args)

    args = ['--folder', directory, '--data', dataset, '--slot_path', slot_definitions, '--slot_train', '--intent_train']
    train_svm(args)

    # train_joint_classifier('--folder', directory, '--data', dataset, '--slot_path', slot_definitions,
    #                        '--trash_intent', no_intent_dataset)

    print('Everything build in {:.0f} seconds'.format(time()-time_start))


if __name__ == '__main__':
    main()
