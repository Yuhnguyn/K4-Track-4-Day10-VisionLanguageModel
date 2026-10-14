"""Control the user-authorized Kaggle interactive session via its Jupyter API.

Connection details live only under ignored work/, never in this source.
"""
import argparse
import datetime
import json
import os
import sys
from pathlib import Path
import uuid
from urllib.parse import urlsplit, urlunsplit, quote

import requests

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def connect():
    status = json.loads(Path(os.environ.get('KAGGLE_SESSION_FILE', 'work/interactive-status.json')).read_text())
    base = status['response']['jupyterUrl'].rstrip('/')
    session = requests.Session()
    response = session.get(base + '/api/kernels', timeout=60)
    response.raise_for_status()
    return base, session, response.json()[0]['id']


def execute(code):
    import websocket
    base, session, kernel = connect()
    split = urlsplit(base + '/api/kernels/' + kernel + '/channels')
    endpoint = urlunsplit(('wss', split.netloc, split.path, split.query, ''))
    ws = websocket.create_connection(endpoint, timeout=60, suppress_origin=True)
    message_id = uuid.uuid4().hex
    header = {'msg_id': message_id, 'username': 'lab', 'session': uuid.uuid4().hex,
              'date': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'msg_type': 'execute_request', 'version': '5.3'}
    ws.send(json.dumps({'header': header, 'parent_header': {}, 'metadata': {}, 'channel': 'shell',
                        'content': {'code': code, 'silent': False, 'store_history': False,
                                    'user_expressions': {}, 'allow_stdin': False,
                                    'stop_on_error': True}}))
    while True:
        message = json.loads(ws.recv())
        if message.get('parent_header', {}).get('msg_id') != message_id:
            continue
        kind, content = message.get('msg_type'), message.get('content', {})
        if kind == 'stream':
            print(content.get('text', ''), end='', flush=True)
        elif kind == 'error':
            raise RuntimeError(content.get('ename', '') + ': ' + content.get('evalue', ''))
        elif kind == 'execute_reply':
            print('Execution status:', content.get('status'))
            break
    ws.close()


def download(remote, local):
    base, session, _ = connect()
    response = session.get(base + '/files/' + quote(remote, safe='/'), timeout=180)
    response.raise_for_status()
    path = Path(local)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(response.content)
    print('Downloaded:', path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--code-file')
    parser.add_argument('--remote')
    parser.add_argument('--local')
    args = parser.parse_args()
    if args.code_file:
        execute(Path(args.code_file).read_text(encoding='utf-8'))
    else:
        download(args.remote, args.local)
