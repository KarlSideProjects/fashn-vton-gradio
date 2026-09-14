"""Real GPU repair of a synthetic shirt blemish; not an anatomy benchmark."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from tryon.config import ROOT, configure
from tryon.engine import Engine


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=Path, default=ROOT / 'examples/model.webp')
    parser.add_argument('--prompt', default='shirt fabric, natural folds, matching the surrounding shirt')
    parser.add_argument('--strength', type=float, default=.99)
    parser.add_argument('--negative-prompt', default='text, logo, letters, pink stain, embroidery')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()
    configure()
    with Image.open(args.image) as image:
        base = image.convert('RGB')
    # Fixed demonstration region, not the application's automatic error detector.
    x, y = int(base.width*.62), int(base.height*.40)
    w, h = max(8, base.width//25), max(8, base.height//40)
    ImageDraw.Draw(base).rectangle((x, y, x+w, y+h), fill='magenta')
    layer = Image.new('RGBA', base.size)
    ImageDraw.Draw(layer).rectangle((x-8, y-8, x+w+8, y+h+8), fill='white')
    editor = {'background': base, 'layers': [layer], 'composite': base}
    png, record, data = Engine().repair(editor, args.prompt, args.strength, args.seed, args.negative_prompt)
    with Image.open(png) as result:
        outside = np.asarray(layer.getchannel('A')) == 0
        np.testing.assert_array_equal(np.asarray(result)[outside], np.asarray(base)[outside])
        assert np.any(np.asarray(result)[~outside] != np.asarray(base)[~outside]), 'No pixel changed inside mask'
    print(json.dumps({'png': png, 'record': record, **data}, ensure_ascii=False, indent=2))
    print('PASS: real GPU inference; selected pixels changed; every outside pixel preserved. Visual QA still required.')


if __name__ == '__main__':
    main()
