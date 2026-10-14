"""Persist Kaggle CLI SDK live logs immediately; credentials stay in the environment."""
import argparse
import json
from pathlib import Path
from kaggle import api

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('kernel')
parser.add_argument('output')
args = parser.parse_args()
path = Path(args.output)
path.parent.mkdir(parents=True, exist_ok=True)
with path.open('w', encoding='utf-8', buffering=1) as handle:
    for event in api.kernels_logs_stream(args.kernel):
        handle.write(json.dumps(event, ensure_ascii=False) + '\n')
        handle.flush()
