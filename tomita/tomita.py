from threading import Lock

import pexpect
from pexpect.exceptions import TIMEOUT

import xmltodict

import json
import os

from concurrent.futures import Future, wait, FIRST_COMPLETED, ThreadPoolExecutor


class Tomita:
    def __init__(self, executable, config, cwd=None, logfile=None):
        assert os.path.isfile(config), 'Config file "{}" not found'.format(config)
        self.name = 'Tomita'
        self.p = pexpect.spawn(executable, [config], cwd=cwd)
        self.p.logfile_read = logfile
        self.p.expect('.* Start.*$')

    def communicate(self, text):
        text = text.strip()
        self.p.sendline(text)
        self.p.expect_exact((text + '\r\n').encode('UTF8'))
        raw = b''
        try:
            while True:
                raw += self.p.read_nonblocking(1, 1)
        except TIMEOUT:
            pass
        return raw.strip() or None

    def get_json(self, text):
        raw = self.communicate(text)
        if not raw:  # empty result
            return []
        if raw.startswith(b'Time:'):  # debug(?) output from tomita, ignore
            return []
        if raw.startswith(b'<document'):
            raw = raw.decode('UTF8').split('\r\n')[0]
            return xmltodict.parse(raw)['document']
        # todo: do something with protobuf and (maybe) text
        raise RuntimeError('Expected xml document, got {}'.format(raw))


class TomitaPool(object):
    def __init__(self, executable, config, cwd=None, logfile=None, num_proc=None):
        if not num_proc:
            import multiprocessing
            num_proc = multiprocessing.cpu_count()
        assert num_proc > 0

        with ThreadPoolExecutor(num_proc) as executor:
            self.pool = [executor.submit(Tomita, executable, config, cwd, logfile) for _ in range(num_proc)]

        self.lock = Lock()

    def get_tomita(self):
        self.lock.acquire()
        wait(self.pool, return_when=FIRST_COMPLETED)
        for i in range(len(self.pool)):
            if self.pool[i].done():
                tomita = self.pool[i].result()
                f = Future()
                self.pool[i] = f
                break
        self.lock.release()
        return tomita, f

    def communicate(self, text):
        print('communicate', text)
        tomita, f = self.get_tomita()
        try:
            res = tomita.communicate(text)
        except:
            f.set_result(tomita)
            raise

        f.set_result(tomita)

        return res

    def get_json(self, text):
        print('get_json', text)
        tomita, f = self.get_tomita()
        res = tomita.get_json(text)

        f.set_result(tomita)

        return res


def main():
    import time

    start = time.time()

    executor = ThreadPoolExecutor(3)

    root = os.path.dirname(os.path.realpath(__file__))
    tomita = TomitaPool(os.environ['TOMITA_PATH'], os.path.join(root, 'test', 'config.proto'),
                        os.path.join(root, 'test'), num_proc=3)

    middle = time.time()

    t = 'ул. Маяковского, c, пятница, 22 апреля 2014 года'

    f1 = executor.submit(tomita.communicate, t)

    f2 = executor.submit(tomita.get_json, t)

    t = 'Юрий Гагарин'

    f3 = executor.submit(tomita.get_json, t)

    wait([f1, f2, f3])

    r = f1.result().decode('UTF8')
    print(r)

    r = f2.result()
    print(json.dumps(r, indent=2, ensure_ascii=False))
    d = r['facts']['Date']
    assert d['DayOfWeek']['@val'] == 'ПЯТНИЦА' and d['Day']['@val'] == '22' and d['Month']['@val'] == 'АПРЕЛЬ', \
        'parsed wrong'

    r = f3.result()
    print(json.dumps(r, indent=2, ensure_ascii=False))
    assert r == [], 'expected empty array'

    end = time.time()
    print('finished in {:.5} seconds ({:.5} for starting and {:.5} for parsing)'
          .format(end - start, middle-start, end-middle))


if __name__ == "__main__":
    main()
