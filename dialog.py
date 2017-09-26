from typing import List
from telegram import User

import logging


class Dialog:
    def __init__(self, preproc_pipeline, nlu_model, policy_model, user: User):
        self.pipeline = preproc_pipeline
        self.nlu_model = nlu_model
        self.policy_model = policy_model
        self.user = user

        self.logger = logging.getLogger('router')
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