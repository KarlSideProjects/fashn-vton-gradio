"""Keep assets and caches beneath the project, before third-party imports."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / 'weights'


def configure():
    for name, path in {
        'HF_HOME': ROOT / '.cache/huggingface',
        'TORCH_HOME': ROOT / '.cache/torch',
        'XDG_CACHE_HOME': ROOT / '.cache',
        'GRADIO_TEMP_DIR': ROOT / '.cache/gradio',
        'TMPDIR': ROOT / '.cache/tmp',
    }.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ['GRADIO_ANALYTICS_ENABLED'] = 'False'
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
