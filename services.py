import json
import urllib.request
from urllib.parse import urlencode


def faq(text: str, threshold=0.95):
    params = {
        "q": text
    }
    url = 'http://lnsigo.dc.phystech.edu:5000/answer?' + urlencode(params)
    response = json.loads(urllib.request.urlopen(url).read().decode('UTF8'))
    response['score'] = round(response['score'], 2)
    answer = response['answer'] if response['score'] >= threshold else None
    return answer, response


def init_chat(chat_id):
    params = {
        "session": chat_id
    }
    url = 'http://lnsigo.dc.phystech.edu:5100/init_session?' + urlencode(params)
    response = json.loads(urllib.request.urlopen(url).read().decode('UTF8'))
    return response


def chat(text: str, chat_id):
    params = {
        "q": text,
        "session": chat_id
    }
    url = 'http://lnsigo.dc.phystech.edu:5100/answer?' + urlencode(params)
    response = json.loads(urllib.request.urlopen(url).read().decode('UTF8'))['answer']
    return response
