import pexpect
from pexpect.exceptions import TIMEOUT

import xmltodict

import json
import os


class Tomita:

    def __init__(self, executable, config, cwd=None, logfile=None):
        assert os.path.isfile(config), 'Config file "{}" not found'.format(config)
        self.name = 'Tomita'
        self.p = pexpect.spawn(executable, [config], cwd=cwd, logfile=logfile)
        self.p.expect('.* Start.*$')

    def communicate(self, text):
        self.p.sendline(text)
        self.p.expect_exact((text + '\r\n').encode('UTF8'))
        raw = b''
        try:
            while True:
                raw += self.p.read_nonblocking(1, 1)
        except TIMEOUT:
            pass
        return raw or None

    def get_json(self, text):
        raw = self.communicate(text)
        if not raw:  # empty result
            return []
        if raw.startswith(b'<document'):
            return xmltodict.parse(raw.decode('UTF8'))['document']
        # todo: do something with protobuf and (maybe) text
        raise RuntimeError('Expected xml document, got {}'.format(raw))


if __name__ == "__main__":
    root = os.path.dirname(os.path.realpath(__file__))
    tomita = Tomita(os.environ['TOMITA_PATH'], os.path.join(root, 'tomita', 'test', 'config.proto'),
                    os.path.join(root, 'tomita', 'test'))

    t = 'ул. Маяковского, c, пятница, 22 апреля 2014 года'

    r = tomita.communicate(t).decode('UTF8')
    print(r)

    r = tomita.get_json(t)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    d = r['facts']['Date']
    assert d['DayOfWeek']['@val'] == 'ПЯТНИЦА' and d['Day']['@val'] == '22' and d['Month']['@val'] == 'АПРЕЛЬ',\
        'parsed wrong'

    t = 'Юрий Гагарин'
    r = tomita.get_json(t)
    print(json.dumps(r, indent=2, ensure_ascii=False))
    assert r == [], 'expected empty array'

    print('end')
