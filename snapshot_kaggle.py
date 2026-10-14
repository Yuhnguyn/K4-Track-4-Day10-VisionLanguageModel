"""Take a bounded live-log snapshot; ending the local reader does not stop Kaggle jobs."""
from concurrent.futures import ThreadPoolExecutor
import json
from pathlib import Path
import subprocess
import sys

if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')


def snapshot(pair):
    slug, filename = pair
    process = subprocess.Popen([sys.executable, 'follow_kaggle.py', slug, filename],
                               stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.terminate()
        process.wait(timeout=10)
    path = Path(filename)
    if not path.exists():
        return {'kernel': slug, 'status': 'No live log received'}
    events = []
    for line in path.read_text(encoding='utf-8').splitlines():
        try:
            events.append(json.loads(line))
        except ValueError:
            pass
    text = ''.join(e.get('data', '') for e in events if e.get('stream_name') == 'stdout')
    lines = [line for line in text.splitlines()
             if ' step ' in line or '] DONE ' in line or '/311 |' in line or ' | F1:' in line]
    return {'kernel': slug, 'progress': lines[-5:]}


if __name__ == '__main__':
    targets = [('nguynhuy9669/k4-track4-day10-vlm-lora-run', 'work/version4-events.jsonl'),
               ('nguynhuy9669/day10-vlm-e3-100', 'work/e3-events.jsonl')]
    with ThreadPoolExecutor(max_workers=2) as pool:
        for result in pool.map(snapshot, targets):
            print(json.dumps(result, ensure_ascii=False))
