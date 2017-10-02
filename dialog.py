from typing import List
from telegram import User

import logging

from policy import GraphBasedSberdemoPolicy
from services import faq, init_chat, chat

from concurrent.futures import ThreadPoolExecutor

import html


class Dialog:
    def __init__(self, preproc_pipeline, nlu_model, policy_model: GraphBasedSberdemoPolicy, user: User,
                 debug=False, patience=3):
        self.pipeline = preproc_pipeline
        self.nlu_model = nlu_model
        self.policy_model = policy_model
        self.user = user

        init_chat(self.user.id)

        self.logger = logging.getLogger('router')
        self.logger.info("{user.id}:{user.name} : started new dialog".format(user=self.user))

        self.debug = debug

        self.executor = ThreadPoolExecutor(max_workers=2)

        self.patience = patience
        self.impatience = 0

    def generate_response(self, client_utterance: str) -> List[str]:
        self.logger.info("{user.id}:{user.name} >>> {msg}".format(user=self.user, msg=repr(client_utterance)))
        message_type = 'text'
        if client_utterance.startswith('__geo__'):
            text = eval(client_utterance.split(' ', 1)[1])
            message_type = 'geo'
        else:
            text = self.pipeline.feed(client_utterance)

        faq_future = self.executor.submit(faq, client_utterance, 0.)
        chat_future = self.executor.submit(chat, client_utterance, self.user.id)

        try:
            nlu_result = self.nlu_model.forward(text, message_type)
        except Exception as e:
            self.logger.error(e)
            return ['NLU ERROR: {}'.format(str(e))]
        self.logger.debug("{user.id}:{user.name} : nlu parsing result: {msg}".format(user=self.user, msg=nlu_result))

        faq_answer, faq_response = faq_future.result()
        self.logger.debug("{user.id}:{user.name} : faq response: `{msg}`".format(user=self.user,
                                                                                 msg=repr(faq_response)))
        chat_response = chat_future.result()
        self.logger.debug("{user.id}:{user.name} : chit-chat response: `{msg}`".format(user=self.user,
                                                                                       msg=repr(chat_response)))

        faq_intent = False
        if faq_answer and faq_answer in self.policy_model.routes:
            faq_intent = True
            if nlu_result.get('intent', 'no_intent') == 'no_intent':
                self.logger.info('using {} intent from faq'.format(faq_answer))
                nlu_result['intent'] = faq_answer

        if not nlu_result['slots']\
                and nlu_result.get('intent', 'no_intent') in ['no_intent', self.policy_model.intent_name]\
                and not faq_answer:
            self.impatience += 1
        else:
            self.impatience = 0

        expect = None
        if faq_answer and not faq_intent:
            response = ["FAQ\n\n" + html.escape(faq_answer)]
        elif self.impatience < self.patience:
            try:
                response, expect = self.policy_model.forward(nlu_result)
                for i in range(len(response)):
                    response[i] = "GOAL-ORIENTED\n" + response[i]
            except Exception as e:
                self.logger.error(e)
                return ['ERROR: {}'.format(str(e))]
            self.nlu_model.set_expectation(expect)
        else:
            response = chat_response  # type:str
            response.replace(', меня зовут', '')
            try:
                names = self.nlu_model.name_parser.parse(chat_response)
                if names:
                    for name in reversed(names):
                        response = response[:name['pos']] + response[name['pos']+name['len']:]
                response = response.strip(' \t\n\r\x0b\x0c,')
                response = response[0].upper() + response[1:]
                response = ["CHIT-CHAT\n" + html.escape(response)]
            except Exception as e:
                self.logger.error(e)
                response = ['CHIT-CHAT ERROR: {}'.format(e)]

        if self.debug:
            debug_message = 'DEBUG\nnlu: {nlu}\n\npolicy: {policy}\n\nfaq: {faq}\n\nchit-chat: {chat}'
            debug_message = debug_message.format(nlu=repr(nlu_result),
                                                 policy=repr({
                                                     'intent_name': self.policy_model.intent_name,
                                                     'slots': self.policy_model.slots,
                                                 }),
                                                 faq=repr({
                                                     'faq_answer': faq_answer,
                                                     'faq_response': faq_response
                                                 }),
                                                 chat=chat_response)
            response.insert(0, html.escape(debug_message))

        for msg in response:
            self.logger.info("{user.id}:{user.name} <<< {msg}".format(user=self.user, msg=repr(msg)))
        self.logger.debug("{user.id}:{user.name} : filled slots: `{msg}`".format(user=self.user,
                                                                                 msg=str(self.policy_model.slots)))
        if expect:
            self.logger.debug("{user.id}:{user.name} : expecting slot `{msg}`".format(user=self.user, msg=expect))

        return response
