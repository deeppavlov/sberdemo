import copy


class GraphBasedSberdemoPolicy(object):

    def __init__(self, routes, slots_objects, sayer):
        self.routes = routes
        self.slots_objects = {s.id: s for s in slots_objects}  # type: Dict[str, DictionarySlot]
        self.sayer = sayer
        self.intent_name = None
        self.intent = None
        self.persistent_slots = {}
        self.slots = {}

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
        if 'intent' in client_nlu and self.intent_name != client_nlu['intent'] and\
                (self.intent is None or client_nlu['intent'] != 'no_intent'):
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

        return responses, expect
