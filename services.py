import json
import urllib.request
from urllib.parse import urlencode


def faq(text: str, threshold=0.95):
    params = {
        "qna_id": "SBRF",
        "q": text
    }
    url = 'http://lnsigo.dc.phystech.edu:5001/answer?' + urlencode(params)
    _, response = json.loads(urllib.request.urlopen(url).read().decode('UTF8'))['answers'][0]
    del response['avg'], response['score']
    response['top'] = round(response['top'], 2)
    for q in response['questions']:
        q['s'] = round(q['s'], 2)
    answer = response['answer'] if response['top'] >= threshold else None
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
