from tomita.tomita import Tomita
import os


def get_value(d, param, default=None):
    if param in d:
        return d[param]['@val'].title()
    return default


class NameParser:

    def __init__(self):
        assert 'TOMITA_PATH' in os.environ, 'Specify path to Tomita binary in $TOMITA_PATH'
        tomita_path = os.environ['TOMITA_PATH']

        root = os.path.dirname(os.path.realpath(__file__))
        cwd = os.path.join(root, 'proper_name')
        config_path = os.path.join(cwd, 'config_names.proto')
        logfile = open(os.path.join(root, '..', 'logs', 'name_parser.log'), 'wb')
        self.tomita = Tomita(tomita_path, config_path, cwd, logfile)

    def parse(self, text):
        if isinstance(text, list):
            text = ' '.join(w['_orig'] for w in text)
        facts = self.tomita.get_json(text.title())

        if not facts:
            return None

        facts = facts['facts']['ProperName']
        if not isinstance(facts, list):
            facts = [facts]

        res = []
        for fact in facts:
            pos = int(fact['@pos'])
            ln = int(fact['@len'])
            name = {
                'pos': pos,
                'len': ln,
                'raw': text[pos:pos+ln].title(),
                'firstname': get_value(fact, 'First'),
                'middlename': get_value(fact, 'Middle'),
                'lastname': get_value(fact, 'Last')
            }
            name['formal'] = name['firstname']
            if name['middlename']:
                name['formal'] += ' ' + name['middlename']
            res.append(name)
        return res
