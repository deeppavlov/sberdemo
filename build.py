import urllib.request
from time import time

from generate_from_templates import main as generate_from_templates
from train_svm import main as train_svm
from train_word_embeddings import main as train_fasttext
from nlu import create_pipe

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
        NO_INTENT = 'sberdemo_no_intent.tsv.gz'
        if not os.path.isfile(NO_INTENT):
            url = 'http://share.ipavlov.mipt.ru:8080/repository/datasets/' + os.path.basename(NO_INTENT)
            try:
                urllib.request.urlretrieve(url, NO_INTENT)
            except:
                pass

        pipe = create_pipe(fasttext_model_path=None)
        with open(no_intent_preprocessed, 'w') as out:
            with gzip.open(NO_INTENT, 'rt', encoding='UTF8') as f:
                for line in f:
                    words = pipe.feed(line.strip())
                    print(' '.join(w['_text'] for w in words), file=out)
        print('{} has been preprocessed'.format(NO_INTENT))

    print()

    args = ['--output', dataset, '--templates', templates]
    generate_from_templates(args)

    args = ['--folder', directory, '--data', dataset, '--slot_path', slot_definitions, '--slot_train', '--oversample']
    train_svm(args)

    args = ['--folder', directory, '--data', dataset, '--slot_path', slot_definitions, '--intent_train', '--oversample']
    train_svm(args)

    print('Everything build in {:.0f} seconds'.format(time()-time_start))


if __name__ == '__main__':
    main()
