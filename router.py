import json
import os
import copy
import time

from nlu import *
from say_actions import Sayer as sayer

from telegram.ext import Updater
from telegram.ext import CommandHandler, MessageHandler, Filters

import threading

from slots import read_slots_serialized


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


class Dialog:
    def __init__(self, preproc_pipeline, nlu_model, policy_model):
        self.pipeline = preproc_pipeline
        self.nlu_model = nlu_model
        self.policy_model = policy_model

    def generate_response(self, client_utterance: str) -> List[str]:
        print('>>>', client_utterance)
        text = self.pipeline.feed(client_utterance)
        try:
            nlu_result = self.nlu_model.forward(text)
        except Exception as e:
            return ['NLU ERROR: {}'.format(str(e))]
        print(nlu_result)
        try:
            response, expect = self.policy_model.forward(nlu_result)
        except Exception as e:
            return ['ERROR: {}'.format(str(e))]
        self.nlu_model.set_expectation(expect)
        print('<<<', response)
        return response


class GraphBasedSberdemoPolicy(object):

    def __init__(self, routes, slots_objects, debug=False):
        self.routes = routes
        self.slots_objects = {s.id: s for s in slots_objects}  # type: Dict[str, DictionarySlot]
        self.intent = None
        self.slots = {}
        self.debug = debug

    def set_intent(self, intent):
        self.slots = dict()
        if not intent:
            self.intent = None
            return
        if intent not in self.routes:
            raise RuntimeError('Unknown intent: ' + str(intent))
        self.intent = copy.deepcopy(self.routes[intent])

    def get_actions(self, tree):
        if not tree:
            return [], False
        actions = []
        done = False
        for i in range(len(tree)):
            branch = tree[i]
            if isinstance(branch, list):
                branch_actions, done = self.get_actions(branch)
                actions += branch_actions
                if done:
                    break
            elif isinstance(branch, dict):
                if 'slot' in branch:
                    if branch['slot'] not in self.slots:
                        if branch.get('not_ask'):
                            break
                        actions.append(['ask', str(branch['slot'])])
                        done = True
                        break
                    slot_filter = self.slots_objects[branch['slot']].filters[branch['condition']]
                    if not slot_filter(self.slots[branch['slot']], branch.get('value')):
                        break
                elif 'action' in branch:
                    if branch.get('executed'):
                        continue
                    branch_actions = [[x.strip() for x in action.split(':')] for action in branch['action'].split(';')
                                      if action]
                    for act, _ in branch_actions:
                        if 'say' != act:
                            done = True
                            break
                    actions += branch_actions
                    branch['executed'] = True
                    if done:
                        break
                else:
                    raise RuntimeError('Node does not have slot nor action')
        return actions, done

    def forward(self, client_nlu):
        print('nlu: ', client_nlu)
        if 'intent' in client_nlu:
            self.set_intent(client_nlu['intent'])
        self.slots.update(client_nlu['slots'])

        actions, _ = self.get_actions(self.intent)
        print(actions)
        if not actions:
            actions = [['say', 'no_intent']]

        expect = None
        responses = []
        for action, value in actions:
            if action == 'ask':
                expect = value
                responses.append(self.slots_objects[value].ask())
            elif action == 'say':
                responses.append(sayer.say(value, self.slots))
            elif action == 'goto':
                if not value:
                    self.intent = None
                    continue
                new_intent_responses, expect = self.forward({"slots": {}, "intent": value})
                responses += new_intent_responses

        if self.debug:
            responses[0] = responses[0] + '\n\nslots: {}'.format(self.slots)

        return responses, expect


def main():
    fname = 'routes.json'
    data = parse_route(fname)

    pipe = create_pipe()

    models_path = './models_nlu'
    slots = read_slots_serialized(models_path, pipe)

    humans = {}

    def new_dialog():
        return Dialog(pipe, StatisticalNLUModel(slots, IntentClassifier(folder=models_path)),
                      GraphBasedSberdemoPolicy(data, slots, debug=True))

    def start(bot, update):
        chat_id = update.message.chat_id
        humans[chat_id] = new_dialog()
        bot.send_message(chat_id=chat_id, text='Здрасте. Чего хотели?')

    def send_delayed(bot, chat_id, messages: list, interval=0.7):
        m = messages.pop(0)
        print('{} <<< {}'.format(chat_id, m))
        bot.send_message(chat_id=chat_id, text=m)
        if messages:
            threading.Timer(interval, send_delayed, [bot, chat_id, messages, interval]).start()

    def user_client(bot, update):

        chat_id = update.message.chat_id
        if chat_id not in humans:
            humans[chat_id] = new_dialog()
        user_msg = update.message.text or str(update.message.location)
        print('{} >>> {}'.format(chat_id, user_msg))
        dialog = humans[chat_id]
        bot_responses = dialog.generate_response(user_msg)
        send_delayed(bot, chat_id, bot_responses, 0.7)

    updater = Updater(token=os.environ['SBER_DEMO_BOT_TOKEN'])
    dispatcher = updater.dispatcher
    start_handler = CommandHandler('start', start)
    msg_handler = MessageHandler(Filters.text | Filters.location, user_client)

    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(msg_handler)

    updater.start_polling()
    updater.idle()


if __name__ == '__main__':
    main()
