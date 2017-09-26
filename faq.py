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
    answer = response['answer'] if response['top'] >= threshold else None
    return answer, response
