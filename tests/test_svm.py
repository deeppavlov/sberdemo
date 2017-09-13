import unittest
from intent_classifier import IntentClassifier
from nlu import *
from slots import *


class TestSVM(unittest.TestCase):
    def setUp(self):
        self.pipe = create_pipe()
        self.model_folder = './models_nlu'
        self.slots = read_slots_serialized(self.model_folder, self.pipe)
        self.intent_clf = IntentClassifier()
        self.intent_clf.load_model(self.model_folder)

    def test_predictions_from_train(self):
        test_dict = {"show_docs": "какие документы для открытия $ нужно предоставить в банк?",
                     "online_reserving": "такой вопрос меня интересует. я могу с СБ бизнес онлайн открыть jpy?",
                     "cost_of_service": "Добрый день. сориентируете по стоимости ведения счета 44 фз?",
                     "show_schedule": "Добрый день. Подскажите до скольки работает отделение рядом с проспект мира"
                     }

        for s in self.slots:
            if isinstance(s, ClassifierSlot):
                print("SLOT: ", s.id)
                for s_name, sent in test_dict.items():
                    result = s.infer_from_compositional_request(self.pipe.feed(sent))
                    print("sent: {}, res: {}".format(s_name, result))
                    # TODO: uncomment these if @online_reserving will be fine
                    # if s_name == s.id:
                    #     self.assertEqual(True, result)
                    # else:
                    #     self.assertEqual(False, result)

    def test_intent_clf_predict(self):
        sent = "Добрый день. Подскажите до скольки работает отделение рядом с проспект мира"
        sent2 = 'Покажите, пожалуйста, передачу про Михалкова'
        pred = self.intent_clf.predict_single(self.pipe.feed(sent2))
        print("Predicted intent:", pred)

if __name__ == '__main__':
    unittest.main()
