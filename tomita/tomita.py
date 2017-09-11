import pexpect
from pexpect.exceptions import TIMEOUT
import os


class Tomita:

    def __init__(self, executable, config):
        self.name = 'Tomita'
        self.p = pexpect.spawn(executable, [config])
        self.p.expect('.* Start.*$')

    def communicate(self, text):
        self.p.sendline(text.encode('UTF8'))
        self.p.expect_exact((text + '\r\n').encode('UTF8'))
        raw = b''
        try:
            while True:
                raw += self.p.read_nonblocking(1, 1)
        except TIMEOUT:
            pass
        return raw or None


if __name__ == "__main__":
    root = os.path.dirname(os.path.realpath(__file__))
    tomita = Tomita(os.path.expanduser('~/Downloads/tomita-linux64'), os.path.join(root, 'config.proto'))
    print(tomita.communicate('ул. Маяковского, пр. Красных Комиссаров, пятница, 22 апреля 2014 года'))
    print(tomita.communicate('Юрий Гагарин'))
    print('end')
