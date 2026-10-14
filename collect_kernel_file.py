"""Download one completed Kaggle artifact by numeric version without listing dataset files."""
import argparse
from pathlib import Path
import requests
from kaggle import api
from kagglesdk.kernels.types.kernels_api_service import ApiDownloadKernelOutputRequest

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('kernel')
parser.add_argument('version', type=int)
parser.add_argument('remote')
parser.add_argument('local')
args = parser.parse_args()
request = ApiDownloadKernelOutputRequest()
request.owner_slug, request.kernel_slug = args.kernel.split('/')
request.version_number = args.version
request.file_path = args.remote
with api.build_kaggle_client() as client:
    try:
        response = client.kernels.kernels_api_client.download_kernel_output(request)
        response.raise_for_status()
        path = Path(args.local)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(response.content)
        print('Downloaded:', path)
    except requests.HTTPError as error:
        print('Artifact not available; HTTP', error.response.status_code)
        raise SystemExit(1)
