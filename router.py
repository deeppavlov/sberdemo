import json
import os
import copy

from nlu import *

from telegram.ext import Updater
from telegram.ext import CommandHandler, MessageHandler, Filters

NEW_ACC_RESERVE_ONLINE = 'NEW_ACC_RESERVE_ONLINE'
NEW_ACC_CURRENCY = 'NEW_ACC_CURRENCY'
CLIENT_RF_RESIDENT = 'CLIENT_RF_RESIDENT'
NEW_ACC_SHOW_DOCS = 'NEW_ACC_SHOW_DOCS'
NEW_ACC_OWNERSHIP_FORM = 'NEW_ACC_OWNERSHIP_FORM'


def format_route(route):
    for i in range(len(route)):
        if isinstance(route[i], list):
            format_route(route[i])
        elif isinstance(route[i], str):
            route[i] = {"slot": route[i], "condition": "any"}
        elif isinstance(route[i], dict):
            if "action" in route[i]:
                pass
            elif len(route[i]) == 1:
                for key, val in route[i].items():
                    route[i] = {"slot": val, "condition": key}


def parse_route(file_name):
    with open(file_name, 'r') as f:
        data = json.load(f)
    for intent, route in data.items():
        format_route(route)

    return data


def some_starts_with(words_list, *prefixi_to_check):
    for w in words_list:
        if w.startswith(prefixi_to_check):
            return True
    return False


class GraphBasedSberdemoNLU:

    def __init__(self):
        self.expect = None

    def set_expectation(self, expect):
        self.expect = expect

    def forward(self, embedding, text: List[Dict]):
        res = {'slots': {}}

        normals = {w['normal'] for w in text}
        intent = None

        slot_values = {NEW_ACC_CURRENCY: None,
                       'slot_address': None,
                       'see_docs': None}

        if {'да', 'давай', 'давайте', 'конечно', 'хорошо', 'ладно', 'ок', 'ok'}.intersection(set(w['_text'] for w in text)):
            res['confirmation'] = True
        if {'нет'}.intersection(set(w['_text'] for w in text)):
            res['confirmation'] = False

        if self.expect and 'confirmation' in res:
            res['slots'][self.expect] = res['confirmation']
            del res['confirmation']

        if {'открыть', 'счет'}.issubset(normals):
            intent = 'new_acc_intent'

        words = [w['_text'] for w in text]
        if some_starts_with(words, 'рубл', 'rub'):
            slot_values[NEW_ACC_CURRENCY] = 'RUB'
        elif some_starts_with(words, 'евро', 'EUR'):
            slot_values[NEW_ACC_CURRENCY] = 'EUR'
        elif some_starts_with(words, 'доллар', 'долар', 'бакс', 'usd', '$', 'dollar'):
            slot_values[NEW_ACC_CURRENCY] = 'USD'

        if some_starts_with(words, 'документ'):
            slot_values[NEW_ACC_SHOW_DOCS] = True

        if some_starts_with(words, 'ooo', 'ооо'):
            slot_values[NEW_ACC_OWNERSHIP_FORM] = 'ооо'
        if some_starts_with(words, 'зао'):
            slot_values[NEW_ACC_OWNERSHIP_FORM] = 'зао'

        if intent is not None:
            res['intent'] = intent
        for k, v in slot_values.items():
            if v:
                res['slots'][k] = v

        return res


class Dialog:
    def __init__(self, preproc_pipeline, nlu_model, policy_model):
        self.pipeline = preproc_pipeline
        self.nlu_model = nlu_model
        self.policy_model = policy_model

    def generate_response(self, client_utterance: str) -> str:
        print('>>>', client_utterance)
        emb, text = self.pipeline.feed(client_utterance)
        nlu_result = self.nlu_model.forward(emb, text)
        print(nlu_result)
        response, expect = self.policy_model.forward(nlu_result)
        self.nlu_model.set_expectation(expect)
        print('<<<', response)
        return response


class GraphBasedSberdemoPolicy(object):

    def __init__(self, routes, slots_objects):
        self.routes = routes
        self.slots_objects = slots_objects
        self.intent = None
        self.slots = dict()

    def set_intent(self, intent):
        self.slots = dict()
        if intent not in self.routes:
            raise RuntimeError('Unknown intent' + str(intent))
        self.intent = copy.deepcopy(self.routes[intent])

    def get_action(self, tree):
        if not tree:
            return None, True

        action = None
        done = False
        to_remove = 0
        for i in range(len(tree)):
            branch = tree[i]
            if isinstance(branch, list):
                action, fork_done = self.get_action(branch)
                if fork_done:
                    to_remove += 1
                if action:
                    break
            elif isinstance(branch, dict):
                if 'slot' in branch:
                    if branch['slot'] not in self.slots:
                        action = 'ask: ' + str(branch['slot'])
                        break
                    # slot_filter = self.slots_objects[branch['slot']].filters[branch['filter']]
                    slot_filter = lambda _, __: True
                    if not slot_filter(self.slots[branch['slot']], branch.get('value')):
                        action = None
                        done = True
                        break
                    continue
                if 'action' in branch:
                    if branch.get('executed'):
                        continue
                    action = branch['action']
                    branch['executed'] = True
                    break
                raise RuntimeError('Node does not have slot nor action')
        del tree[:to_remove]
        if action is None:
            done = True
        return action, done

    def forward(self, client_nlu):
        print('nlu: ', client_nlu)
        if 'intent' in client_nlu:
            self.set_intent(client_nlu['intent'])
        self.slots.update(client_nlu['slots'])

        actions, done = self.get_action(self.intent)
        if not actions:
            actions = 'say: no intent'
        actions = [[x.strip() for x in action.split(':')] for action in actions.split(';') if action]

        expect = None
        for action, value in actions:
            if action == 'ask':
                expect = value

        return 'Action: ' + str(actions) + '; done: ' + str(done), expect


def main():
    fname = 'new_acc.json'
    data = parse_route(fname)

    pipe = Pipeline(sent_tokenize, word_tokenize, [PyMorphyPreproc(), Lower()], embedder=np.vstack)

    humans = {}

    def start(bot, update):
        chat_id = update.message.chat_id
        humans[chat_id] = Dialog(pipe, GraphBasedSberdemoNLU(), GraphBasedSberdemoPolicy(data, None))
        bot.send_message(chat_id=chat_id, text='Здрасте. Чего хотели?')

    def user_client(bot, update):

        chat_id = update.message.chat_id
        user_msg = update.message.text
        print('{} >>> {}'.format(chat_id, user_msg))
        dialog = humans[chat_id]
        bot_resp = dialog.generate_response(user_msg)
        print('{} <<< {}'.format(chat_id, bot_resp))
        bot.send_message(chat_id=chat_id, text=bot_resp)


    updater = Updater(token=os.environ['SBER_DEMO_BOT_TOKEN'])
    dispatcher = updater.dispatcher
    start_handler = CommandHandler('start', start)
    msg_handler = MessageHandler(Filters.text, user_client)

    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(msg_handler)

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
