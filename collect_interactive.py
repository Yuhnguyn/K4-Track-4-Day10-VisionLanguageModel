"""Collect measured results from the authorized Kaggle session without waiting for Run All."""
import json
from pathlib import Path
from urllib.parse import quote
import requests
from kaggle_jupyter import connect


def main():
    base, session, _ = connect()
    downloaded = []

    def fetch(remote, local):
        response = session.get(base + '/files/' + quote(remote, safe='/'), timeout=180)
        if response.status_code == 404:
            return
        response.raise_for_status()
        local = Path(local)
        if local.exists() and local.read_bytes() == response.content:
            return
        local.parent.mkdir(parents=True, exist_ok=True)
        local.write_bytes(response.content)
        downloaded.append(str(local))

    def directory(relative):
        remote = 'day10-lab/' + relative
        response = session.get(base + '/api/contents/' + quote(remote, safe='/'), timeout=60)
        if response.status_code == 404:
            return
        response.raise_for_status()
        for item in response.json().get('content', []):
            if item['type'] == 'directory':
                directory(relative + '/' + item['name'])
            elif Path(item['name']).suffix in ('.json', '.jsonl', '.png'):
                fetch(item['path'], relative + '/' + item['name'])

    for relative in ('report/evidence', 'report/figures'):
        directory(relative)
    for tag in ('e1', 'e2', 'e1_d10'):
        fetch(f'day10-lab/work/{tag}/run.json', f'work/{tag}/run.json')
        fetch(f'day10-lab/work/{tag}.jsonl', f'work/{tag}.jsonl')
    for log in ('fast-pair.log', 'e2-worker-fixed.log', 'e4-watch.log', 'e4-worker-live.log'):
        fetch(log, 'work/' + log)
    print(json.dumps({'updated': downloaded}, ensure_ascii=False))


if __name__ == '__main__':
    main()
