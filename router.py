import json
import os
import copy
import logging

from telegram import Update, User, Bot

from nlu import *
from say_actions import Sayer

from telegram.ext import Updater
from telegram.ext import CommandHandler, MessageHandler, Filters

import threading

from slots import read_slots_serialized
from tomita.name_parser import NameParser


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
    def __init__(self, preproc_pipeline, nlu_model, policy_model, user: User):
        self.pipeline = preproc_pipeline
        self.nlu_model = nlu_model
        self.policy_model = policy_model
        self.user = user

        self.logger = get_logger()
        self.logger.info("{user.id}:{user.name} : started new dialog".format(user=self.user))

    def generate_response(self, client_utterance: str) -> List[str]:
        self.logger.info("{user.id}:{user.name} >>> {msg}".format(user=self.user, msg=repr(client_utterance)))
        message_type = 'text'
        if client_utterance.startswith('__geo__'):
            text = eval(client_utterance.split(' ', 1)[1])
            message_type = 'geo'
        else:
            text = self.pipeline.feed(client_utterance)

        try:
            nlu_result = self.nlu_model.forward(text, message_type)
        except Exception as e:
            self.logger.error(e)
            return ['NLU ERROR: {}'.format(str(e))]
        self.logger.debug("{user.id}:{user.name} : nlu parsing result: {msg}".format(user=self.user, msg=nlu_result))
        try:
            response, expect = self.policy_model.forward(nlu_result)
        except Exception as e:
            self.logger.error(e)
            return ['ERROR: {}'.format(str(e))]
        self.nlu_model.set_expectation(expect)
        for msg in response:
            self.logger.info("{user.id}:{user.name} <<< {msg}".format(user=self.user, msg=repr(msg)))
        self.logger.debug("{user.id}:{user.name} : filled slots: `{msg}`".format(user=self.user,
                                                                                 msg=str(self.policy_model.slots)))
        if expect:
            self.logger.debug("{user.id}:{user.name} : expecting slot `{msg}`".format(user=self.user, msg=expect))
        return response


class GraphBasedSberdemoPolicy(object):

    def __init__(self, routes, slots_objects, sayer, debug=False):
        self.routes = routes
        self.slots_objects = {s.id: s for s in slots_objects}  # type: Dict[str, DictionarySlot]
        self.sayer = sayer
        self.intent_name = None
        self.intent = None
        self.persistent_slots = {}
        self.slots = {}
        self.debug = debug

    def set_intent(self, intent):
        self.intent_name = intent or None
        self.slots = copy.deepcopy(self.persistent_slots)
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
        if 'intent' in client_nlu:
            self.set_intent(client_nlu['intent'])

        self.slots.update(client_nlu['slots'])

        actions, _ = self.get_actions(self.intent)
        if not actions:
            actions = [['say', 'no_intent']]

        if 'name' in client_nlu and client_nlu['name']:
            self.persistent_slots['client_name'] = client_nlu['name']
            self.slots['client_name'] = client_nlu['name']

            for i in range(len(actions)):
                if actions[i] == ['say', 'no_intent']:
                    actions[i] = ['say', 'no_intent_named']

        expect = None
        responses = []
        for action, value in actions:
            if action == 'ask':
                expect = value
                responses.append(self.slots_objects[value].ask())
            elif action == 'say':
                responses.append(self.sayer.say(value, self.slots))
            elif action == 'goto':
                if not value:
                    self.intent = None
                    continue
                new_intent_responses, expect = self.forward({"slots": {}, "intent": value})
                responses += new_intent_responses

        if self.debug:
            responses[0] = responses[0] + '\n\nslots: {}'.format(self.slots)

        return responses, expect


def set_logger(level=logging.DEBUG):
    logger = logging.getLogger('router')
    logger.setLevel(level)

    fh = logging.FileHandler(os.path.join('.', 'logs', 'router.log'))
    fh.setLevel(level)

    ch = logging.StreamHandler()
    ch.setLevel(level)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    ch.setFormatter(formatter)

    logger.addHandler(fh)
    logger.addHandler(ch)


def get_logger():
    return logging.getLogger('router')


def main():
    set_logger()
    get_logger().info('Starting...')

    fname = 'routes.json'
    data = parse_route(fname)

    pipe = create_pipe()

    models_path = './models_nlu'
    slots = read_slots_serialized(models_path, pipe)

    name_parser = NameParser()

    sayer = Sayer(slots, pipe)

    humans = {}

    def new_dialog(user):
        debug = True
        return Dialog(pipe, StatisticalNLUModel(slots, IntentClassifier(folder=models_path), name_parser),
                      GraphBasedSberdemoPolicy(data, slots, sayer, debug=debug), user)

    def start(bot: Bot, update: Update):
        chat_id = update.message.chat_id
        try:
            humans[chat_id] = new_dialog(update.effective_user)
        except Exception as e:
            get_logger().error(e)
        bot.send_message(chat_id=chat_id, text=sayer.say('greeting', {}))

    def send_delayed(bot: Bot, chat_id, messages: list, interval=0.7):
        m = messages.pop(0)
        bot.send_message(chat_id=chat_id, text=m, parse_mode='HTML')
        if messages:
            threading.Timer(interval, send_delayed, [bot, chat_id, messages, interval]).start()

    def user_client(bot: Bot, update):

        chat_id = update.message.chat_id
        if chat_id not in humans:
            humans[chat_id] = new_dialog(update.effective_user)
        user_msg = update.message.text or '__geo__ ' + str(update.message.location)
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

    get_logger().info('Ready')

    updater.idle()


if __name__ == '__main__':
    main()
