"""Archive live measured stdout as checkpoints, never as official scorer evidence."""
import json
from pathlib import Path
import re

path = Path('work/version4-events.jsonl')
events = [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines()]
stdout = ''.join(event.get('data', '') for event in events if event.get('stream_name') == 'stdout')
out = Path('report/checkpoints')
for name in ('E1', 'E2', 'E3'):
    pattern = re.compile(r'\[' + name + r'\] DONE (\{[^\n]+\})')
    records = pattern.findall(stdout)
    if not records:
        continue
    run = json.loads(records[-1])
    summary_lines = [line for line in stdout.splitlines()
                     if line.startswith(name + ' [test]') or line.startswith(name + ' | F1:')]
    record = {'source': 'Kaggle live stdout from k4-track4-day10-vlm-lora-run version 4',
              'status': 'Training complete; test summary available' if summary_lines else 'Training complete',
              'training': run, 'scorer_stdout_summary': summary_lines[-2:],
              'limitation': 'Scorer stdout rounds metrics. This checkpoint is not report/evidence/'
                            + name.lower() + '.json and does not replace the scorer output file.'}
    (out / ('04-' + name.lower() + '-kaggle-progress.json')).write_text(
        json.dumps(record, indent=2, ensure_ascii=False), encoding='utf-8')
    print(name, record['status'])
lines = [line for line in stdout.splitlines()
         if re.match(r'\[E[123]\]|E[123] \[test\]|E[123] \| F1:', line)]
(out / '04-kaggle-training-progress.log').write_text('\n'.join(lines) + '\n', encoding='utf-8')
