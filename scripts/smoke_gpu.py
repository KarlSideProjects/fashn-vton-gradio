"""Real model inference only: missing GPU/weights raises an error, never skips."""
import json
import os

from tryon.config import ROOT, configure


def main():
    configure()
    os.environ['HF_HUB_OFFLINE'] = '1'
    from PIL import Image
    from tryon.engine import Engine
    engine = Engine()
    with Image.open(ROOT / 'examples/model.webp') as person, Image.open(ROOT / 'examples/garment.webp') as garment:
        for index in range(2):
            png, report, data = engine.generate(person, garment, 'tops', 'model', 30, 42)
            print(json.dumps({'run': index + 1, 'png': png, 'report': report, **data}, ensure_ascii=False, indent=2))
    print('Two real GPU generations completed (cold/warm). Inspect both PNGs for garment and identity quality.')


if __name__ == '__main__':
    main()
