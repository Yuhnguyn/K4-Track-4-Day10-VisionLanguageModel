"""Prepare E1/E2 on the two GPUs of the separately authorized interactive session."""
import json
from pathlib import Path

notebook = json.loads(Path('work/kaggle-e1-1/run.ipynb').read_text(encoding='utf-8'))
bootstrap, execution = [''.join(cell['source']) for cell in notebook['cells']]
imports = 'import os, sys, subprocess\nfrom pathlib import Path\n' + execution[:execution.index("print('GPU:")]
worker = execution[execution.index('config = yaml.safe_load'):]
worker = worker.replace('configs/e1.yaml', 'configs/e2.yaml').replace("tag = 'e1' if", "tag = 'e2' if")
insertion = f'''
Path('e2_worker.py').write_text({(imports + worker)!r})
second_log = open('/kaggle/working/e2-worker.log', 'w')
second = subprocess.Popen([sys.executable, 'e2_worker.py'],
                         env=dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYTHONUNBUFFERED='1'),
                         stdout=second_log, stderr=subprocess.STDOUT)
print('Independent E2 on GPU1 started', flush=True)
'''
execution = execution.replace('config = yaml.safe_load', insertion + 'config = yaml.safe_load', 1)
execution += "\nprint('E2 exit code:', second.wait(), flush=True)\nsecond_log.close()\n"
program = bootstrap + execution
launcher = f'''from pathlib import Path
import subprocess, sys, os
Path('/kaggle/working/fast_pair.py').write_text({program!r})
log = open('/kaggle/working/fast-pair.log', 'w')
process = subprocess.Popen([sys.executable, '-u', '/kaggle/working/fast_pair.py'],
                           stdout=log, stderr=subprocess.STDOUT)
print('Independent E1/E2 process launched:', process.pid)
'''
Path('work/start-pair.py').write_text(launcher, encoding='utf-8')
print('Prepared interactive E1/E2 launcher')
