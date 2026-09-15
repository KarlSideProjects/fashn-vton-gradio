"""Explicit downloads, local cache, revision and checksum record."""
import hashlib
import json
import os
from pathlib import Path

from tryon.config import ROOT, WEIGHTS, configure


def sha256(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def main():
    configure()
    os.environ['HF_HUB_OFFLINE'] = '0'
    from huggingface_hub import HfApi, hf_hub_download
    WEIGHTS.mkdir(parents=True, exist_ok=True)
    manifest_path = WEIGHTS / 'manifest.json'
    previous = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    revisions = previous.get('revisions', {})
    api = HfApi()
    files = {}
    for repo, names, target in [
        ('fashn-ai/fashn-vton-1.5', ['model.safetensors'], WEIGHTS),
        ('fashn-ai/DWPose', ['yolox_l.onnx', 'dw-ll_ucoco_384.onnx'], WEIGHTS / 'dwpose'),
    ]:
        revision = revisions.get(repo) or api.model_info(repo).sha
        revisions[repo] = revision
        for name in names:
            print(f'Downloading {repo}/{name} @ {revision}', flush=True)
            path = Path(hf_hub_download(repo, name, revision=revision, local_dir=target))
            files[str(path.relative_to(ROOT))] = sha256(path)
    # Official parser API caches its own weights in the project-local HF_HOME.
    from fashn_human_parser import FashnHumanParser
    FashnHumanParser(device='cpu')
    # Record snapshots used by the parser; model inference is tested separately.
    parser_cache_files = {}
    for path in (ROOT / '.cache/huggingface/hub').glob('models--*/snapshots/*/**/*'):
        if path.is_file():
            parser_cache_files[str(path.relative_to(ROOT))] = sha256(path)
    data = {'revisions': revisions, 'sha256': files, 'parser_cache_sha256': parser_cache_files}
    temporary = WEIGHTS / 'manifest.json.tmp'
    temporary.write_text(json.dumps(data, indent=2), encoding='utf-8')
    temporary.replace(manifest_path)
    print('Downloaded and checksummed weights. Inference has not been tested by this command.')


if __name__ == '__main__':
    main()
