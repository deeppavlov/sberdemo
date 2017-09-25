from time import time

from generate_from_templates import main as generate_from_templates
from train_svm import main as train_svm
from train_word_embeddings import main as train_fasttext
from nlu import create_pipe

import os
import shutil


def main():
    time_start = time()
    root = os.path.dirname(os.path.realpath(__file__))
    models_dir = 'models_nlu'
    directory = os.path.join(root, models_dir)
    fasttext_model = os.path.join(root, 'fasttext.sber.bin')

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
        pipe = create_pipe(fasttext_model_path=None)
        with open(no_intent_preprocessed, 'w') as out:
            with open('no_intent.tsv') as f:
                for line in f:
                    words = pipe.feed(line.strip())
                    print(' '.join(w['_text'] for w in words), file=out)
        print('no_intent.tsv has been preprocessed')

    print()

    print('training FastText...')
    train_fasttext('--fasttext_model', fasttext_model, '--dataset_file', no_intent_preprocessed)
    print('FastText trained')

    templates = 'generative_templates.tsv'
    slot_definitions = 'slots_definitions.tsv'

    args = ['--output', dataset, '--templates', templates]
    generate_from_templates(args)

    args = ['--folder', directory, '--data', dataset, '--slot_path', slot_definitions, '--slot_train', '--oversample']
    train_svm(args)

    args = ['--folder', directory, '--data', dataset, '--slot_path', slot_definitions, '--intent_train', '--oversample']
    train_svm(args)

    print('Everything build in {:.0f} seconds'.format(time()-time_start))


if __name__ == '__main__':
    main()
