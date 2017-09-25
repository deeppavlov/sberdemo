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
        res = self.tomita.get_json(text.title())
        if not res:
            return None
        res = res['facts']['ProperName']
        pos = int(res['@pos'])
        ln = int(res['@len'])
        res = {
            'raw': text[pos:pos+ln].title(),
            'firstname': get_value(res, 'First'),
            'middlename': get_value(res, 'Middle'),
            'lastname': get_value(res, 'Last')
        }
        res['formal'] = res['firstname']
        if res['middlename']:
            res['formal'] += ' ' + res['middlename']
        return res
