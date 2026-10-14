"""Package the current lab source as a self-contained private Kaggle CLI kernel.

Usage: python prepare_kaggle.py --username ACCOUNT --experiment e1
Then: kaggle kernels push -p work/kaggle-e1
"""
import argparse
import base64
import io
import json
from pathlib import Path
import zipfile


def build(username, experiment, fraction=1.0, with_e4=False):
    root = Path(__file__).resolve().parent
    output = root / 'work' / f'kaggle-{experiment}-{fraction:g}'
    output.mkdir(parents=True, exist_ok=True)
    archive = io.BytesIO()
    with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as z:
        files = (list(root.glob('student_*.py')) + list(root.glob('*.md'))
                 + [root / 'requirements-colab.txt', root / 'lab_day10.ipynb', root / 'pytest.ini'])
        for directory in ('scripts', 'configs', 'tests', 'student_checks', 'report/evidence'):
            files += [p for p in (root / directory).rglob('*')
                      if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
        for path in files:
            z.write(path, path.relative_to(root).as_posix())
    payload = base64.b64encode(archive.getvalue()).decode()
    bootstrap = f'''import base64, io, os, zipfile, subprocess, sys
from pathlib import Path
ROOT = Path('/kaggle/working/day10-lab')
ROOT.mkdir(parents=True, exist_ok=True)
zipfile.ZipFile(io.BytesIO(base64.b64decode({payload!r}))).extractall(ROOT)
os.chdir(ROOT)
sys.path.insert(0, str(ROOT))
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements-colab.txt'], check=True)
subprocess.run([sys.executable, '-m', 'pip', 'uninstall', '-y', '-q', 'torchao'], check=False)
'''
    execution = f'''
import gc, json, shutil, time, traceback
import torch, yaml
from transformers import set_seed
from scripts.labcommon import read_jsonl
from student_data import explore
from student_labels import verify_coordinates
from student_inference import load_base_model, generate_predictions
from student_train import train_setup, load_adapter_and_projector
from student_figures import build as figures

print('GPU:', torch.cuda.get_device_name(0), flush=True)
sources = [p.parent for p in Path('/kaggle/input').rglob('labels-YOLO')
           if (p.parent / 'images').is_dir()]
assert sources, 'Dataset source not found'
subprocess.run([sys.executable, 'scripts/prepare_data.py', '--source', str(sources[0])], check=True)
explore()
verify_coordinates()
subprocess.run([sys.executable, '-m', 'pytest', '-q', 'tests', 'student_checks',
                '-k', 'not report_template_has_student_markers_and_ten_questions'], check=True)
config = yaml.safe_load(Path('configs/{experiment}.yaml').read_text())
set_seed(config['seed'])
tag = '{experiment}' if {fraction!r} == 1.0 else '{experiment}_d10'
out = Path('work') / tag
try:
    model, processor = load_base_model(config['model_id'])
    model, info = train_setup(model, processor, config, read_jsonl('data/splits/train.jsonl'),
                              'data', out, fraction={fraction!r})
    del model, processor
    gc.collect()
    torch.cuda.empty_cache()
    base, processor = load_base_model(config['model_id'])
    model, count = load_adapter_and_projector(base, out)
    print('Reloaded projector tensors:', count, flush=True)
    pred = Path('work') / (tag + '.jsonl')
    stats = generate_predictions(model, processor, read_jsonl('data/splits/test.jsonl'),
                                 'data', pred, batch_size=12)
    (out / 'inference.json').write_text(json.dumps(stats, indent=2))
    evidence = 'report/evidence/' + (tag if {fraction!r} == 1.0 else 'e4_data_size_0.1') + '.json'
    subprocess.run([sys.executable, 'scripts/score.py', '--pred', str(pred), '--name', tag.upper(),
                    '--out', evidence, '--meta', 'trainable_params=' + str(info['trainable_params']),
                    '--meta', 'peak_vram_gb=' + str(info['peak_vram_gb']),
                    '--meta', 'train_minutes=' + str(info['train_minutes'])], check=True)
    figures(str(pred), prefix=tag)
except Exception:
    Path('/kaggle/working/failure.txt').write_text(traceback.format_exc())
    raise
finally:
    dest = Path('/kaggle/working/results')
    dest.mkdir(exist_ok=True)
    shutil.copytree('report', dest / 'report', dirs_exist_ok=True)
    if Path('work').exists():
        shutil.copytree('work', dest / 'work', dirs_exist_ok=True)
    shutil.make_archive('/kaggle/working/day10-results-' + tag, 'zip', dest)
'''
    if with_e4:
        # Each Kaggle T4 session has two GPUs. Independent processes avoid model/optimizer sharing.
        worker = execution[execution.index("config = yaml.safe_load"):]
        worker = worker.replace("configs/e3.yaml", "configs/e1.yaml")
        worker = worker.replace("tag = 'e3' if 1.0 == 1.0 else 'e3_d10'", "tag = 'e1_d10'")
        worker = worker.replace("fraction=1.0", "fraction=0.1")
        worker = worker.replace("tag if 1.0 == 1.0 else 'e4_data_size_0.1'", "'e4_data_size_0.1'")
        imports = 'import os, sys, subprocess\nfrom pathlib import Path\n' + execution[:execution.index("print('GPU:")]
        insertion = f'''\nPath('e4_worker.py').write_text({(imports + worker)!r})
e4_log = open('/kaggle/working/e4-worker.log', 'w')
e4_environment = dict(os.environ, CUDA_VISIBLE_DEVICES='1', PYTHONUNBUFFERED='1')
e4_process = subprocess.Popen([sys.executable, 'e4_worker.py'], env=e4_environment,
                              stdout=e4_log, stderr=subprocess.STDOUT)
print('E4 E1/data10 started on GPU1; E3 on GPU0', flush=True)
'''
        execution = execution.replace("config = yaml.safe_load", insertion + "config = yaml.safe_load", 1)
        execution += '''\nprint('Waiting for E4 on GPU1', flush=True)
e4_code = e4_process.wait()
e4_log.close()
print('E4 exit code:', e4_code, flush=True)
assert e4_code == 0, Path('/kaggle/working/e4-worker.log').read_text()[-5000:]
'''
    notebook = {'nbformat': 4, 'nbformat_minor': 5,
                'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
                'cells': [{'id': f'cell-{index}', 'cell_type': 'code', 'metadata': {}, 'execution_count': None,
                           'outputs': [], 'source': source.splitlines(keepends=True)}
                          for index, source in enumerate((bootstrap, execution))]}
    (output / 'run.ipynb').write_text(json.dumps(notebook), encoding='utf-8')
    metadata = {'id': f'{username}/day10-vlm-{experiment}-{int(fraction * 100)}',
                'title': f'day10-vlm-{experiment}-{int(fraction * 100)}',
                'code_file': 'run.ipynb', 'language': 'python', 'kernel_type': 'notebook',
                'is_private': True, 'enable_gpu': True, 'enable_internet': True,
                'dataset_sources': ['lorenzoarcioni/road-damage-dataset-potholes-cracks-and-manholes'],
                'competition_sources': [], 'kernel_sources': []}
    (output / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
    print(output)
    return output


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--username', required=True)
    parser.add_argument('--experiment', choices=['e1', 'e2', 'e3'], required=True)
    parser.add_argument('--fraction', type=float, choices=[0.1, 1.0], default=1.0)
    parser.add_argument('--with-e4', action='store_true', help='Run E1/data10 on second T4 alongside E3')
    args = parser.parse_args()
    build(args.username, args.experiment, args.fraction, args.with_e4)
