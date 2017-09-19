from generate_from_templates import main as generate_from_templates
from train_svm import main as train_svm

import os
import shutil


def main():
    root = os.path.dirname(os.path.realpath(__file__))
    directory = os.path.join(root, 'models_nlu')

    dataset = os.path.join(root, 'generated_dataset.tsv')

    shutil.rmtree(directory, ignore_errors=True)
    try:
        os.remove(dataset)
    except OSError:
        pass

    os.mkdir(directory)

    templates = 'generative_templates.tsv'
    slot_definitions = 'slots_definitions.tsv'

    args = ['--output', dataset, '--templates', templates]
    generate_from_templates(args)

    args = ['--folder', directory, '--data', dataset, '--slot_path', slot_definitions, '--slot_train', '--oversample']
    train_svm(args)

    args = ['--folder', directory, '--data', dataset, '--slot_path', slot_definitions, '--intent_train', '--oversample']
    train_svm(args)


if __name__ == '__main__':
    main()
