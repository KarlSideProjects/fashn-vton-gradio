"""Run the actual Space-style app once; never substitutes a mock image."""
import argparse
import json
import time
from uuid import uuid4

from PIL import Image

from app import try_on
from tryon.config import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--person', default=str(ROOT / 'examples/model.webp'))
    parser.add_argument('--garment', default=str(ROOT / 'examples/garment.webp'))
    parser.add_argument('--device', choices=['cpu', 'cuda'], default='cpu')
    args = parser.parse_args()
    started = time.perf_counter()
    with Image.open(args.person) as person, Image.open(args.garment) as garment:
        result, status = try_on(person, garment, 'tops', 'model', 30, 1.5, 42, True, args.device)
    if result is None:
        raise RuntimeError(status)
    directory = ROOT / 'outputs' / 'space-smoke' / uuid4().hex
    directory.mkdir(parents=True, exist_ok=False)
    result.save(directory / 'result.png')
    report = dict(status=status, seconds=round(time.perf_counter() - started, 3),
                  device=args.device, noise='cpu_fp32_v1', steps=30, guidance_scale=1.5, seed=42,
                  category='tops', garment_photo_type='model', segmentation_free=True,
                  postprocessing='none', result_size=result.size,
                  person=args.person, garment=args.garment, output=str(directory / 'result.png'))
    (directory / 'report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2), flush=True)
    print('PASS: real generation. Inspect the image; completion is not a quality guarantee.')


if __name__ == '__main__':
    main()
