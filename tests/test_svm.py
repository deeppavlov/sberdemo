import unittest
from intent_classifier import IntentClassifier
from nlu import *
from slots import *
import pandas as pd
from time import time


class TestSVM(unittest.TestCase):
    def setUp(self):
        self.pipe = create_pipe()
        self.model_folder = './models_nlu'
        self.slots = read_slots_serialized(self.model_folder, self.pipe)
        self.intent_clf = IntentClassifier(folder=self.model_folder)
        self.table = pd.read_csv('generated_dataset.tsv', sep='\t')

    def _test_binary_clf(self, predict_single_func, labels, key, map_for_data=None):
        count = 0
        right = 0
        res_counter = {}
        for k in labels:
            res_counter.setdefault(k, defaultdict(int))
        for row_id, row in self.table[['request', key]].iterrows():
            text = row['request']
            if map_for_data is not None:
                value = map_for_data[row[key]]
            else:
                value = row[key]
            pred = predict_single_func(self.pipe.feed(text))
            count += 1
            if value == pred:
                right += 1
                res_counter[value]["TP"] += 1
            else:
                res_counter[value]["FN"] += 1
                res_counter[pred]["FP"] += 1
                print("TRUE: <{}>, but PRED: <{}> for {}".format(value, pred, text))

        print("Overall accuracy: {}".format(right / count))
        print("--Per classes--")
        for k, val in res_counter.items():
            print("class: {}".format(k))
            print("recall: {}".format(val["TP"] / (val["TP"] + val["FN"])))
            print("precision: {}".format(val["TP"] / (val["TP"] + val["FP"])))
            print("overall class examples in the data: {}".format(val["TP"] + val["FN"]))
            print("-" * 50)
        return right / count

    # TODO: update test for intent_clf
    # def test_clf_intent(self):
    #     print("== Testing Intent clf ==")
    #     labels = list(set(self.table["intent"]))
    #     overall_acc = self._test_binary_clf(predict_single_func=self.intent_clf.predict_single,
    #                                         labels=labels,
    #                                         key='intent')
    #     self.assertGreater(overall_acc, 0.95)

    def test_clf_slots(self):
        print("== Testing slots clf ==")
        time_started = time()
        for slot in self.slots:
            if isinstance(slot, ClassifierSlot):
                print("\n** Testing <{}> slot **\n".format(slot.id))
                overall_acc = self._test_binary_clf(predict_single_func=slot.infer_from_compositional_request,
                                                    labels=[None, slot.true],
                                                    key=slot.id,
                                                    map_for_data={np.nan: None, 'E': slot.true, None:None})
                self.assertGreater(overall_acc, 0.7)

        print('\t\t{:.2f} seconds'.format(time() - time_started))

if __name__ == '__main__':
    unittest.main()
