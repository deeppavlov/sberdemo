import csv
import re
from itertools import chain


def main():
    words = set()
    with open('rus_voc.txt') as f:
        for line in f:
            words.add(line.strip())

    words_in_slots = []
    with open('slots_definitions.tsv') as f:
        csv_rows = csv.reader(f, delimiter='\t', quotechar='"')
        for cell in chain(*csv_rows):
            if (not cell.strip()) or ('->' in cell) or ('Error' in cell):
                continue
            cell_value = cell.strip().replace(', ', ',')
            print(cell_value)

            for w in re.split('\W+', cell_value):
                w = w.lower().strip()
                if w:
                    words_in_slots.append(w)

    with open('sber_voc.txt', 'w') as f:
        for w in sorted(words.union(set(words_in_slots))):
            print(w, file=f)

if __name__ == '__main__':
    main()




