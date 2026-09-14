"""Real masking/compositing tests; the worker fixture does not validate GPU quality."""
import json
import importlib.util
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image

from tryon.config import configure
from tryon.repair import prepare_repair, crop_box, composite_patch, run_repair


class RepairTests(unittest.TestCase):
    def setUp(self):
        configure()
        self.base = Image.new('RGB', (180, 120), 'blue')
        self.layer = Image.new('RGBA', self.base.size)
        self.layer.paste((255, 255, 255, 255), (20, 30, 40, 60))
        self.editor = {'background': self.base, 'layers': [self.layer],
                       'composite': Image.new('RGB', self.base.size, 'white')}

    def test_uses_original_background_and_layer_alpha_not_painted_composite(self):
        base, mask, options = prepare_repair(self.editor, ' fabric ', .75, 42)
        self.assertEqual(base.getpixel((25, 35)), (0, 0, 255))
        self.assertEqual(mask.getbbox(), (20, 30, 40, 60))
        self.assertEqual(options['prompt'], 'fabric')

    def test_mask_and_parameter_validation(self):
        cases = [
            (None, 'x', .75, 1),
            ({**self.editor, 'layers': []}, 'x', .75, 1),
            ({**self.editor, 'layers': [Image.new('RGBA', self.base.size, 'white')]}, 'x', .75, 1),
            ({**self.editor, 'layers': [Image.new('RGBA', (100, 100))]}, 'x', .75, 1),
            ({**self.editor, 'layers': [self.base]}, 'x', .75, 1),
            (self.editor, '', .75, 1), (self.editor, 'x', float('nan'), 1),
            (self.editor, 'x', 1, 1), (self.editor, 'x', .75, True),
            (self.editor, 'x', .75, 1.5),
        ]
        for args in cases:
            with self.subTest(args=args[1:]), self.assertRaises(ValueError):
                prepare_repair(*args)

    def test_composite_changes_only_selected_pixels_including_at_image_edge(self):
        for box in [(0, 0, 10, 10), (20, 30, 40, 60), (170, 110, 180, 120)]:
            mask = Image.new('L', self.base.size)
            mask.paste(255, box)
            crop = crop_box(mask)
            filled = Image.new('RGB', (crop[2]-crop[0], crop[3]-crop[1]), 'red')
            result = composite_patch(self.base, mask, filled, crop)
            outside = np.asarray(mask) == 0
            np.testing.assert_array_equal(np.asarray(result)[outside], np.asarray(self.base)[outside])
            self.assertEqual(result.getpixel(box[:2]), (255, 0, 0))
            self.assertEqual(self.base.getpixel(box[:2]), (0, 0, 255))

    def test_worker_success_publishes_reproducible_bundle(self):
        def worker(command, **kwargs):
            work = Path(command[-1])
            job = json.loads((work/'job.json').read_text())
            box = job['crop_box']
            Image.new('RGB', (box[2]-box[0], box[3]-box[1]), 'red').save(work/'patch.png')
            (work/'metrics.json').write_text('{"quality": "test fixture"}')
            return SimpleNamespace(returncode=0)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'manifest.json').write_text('{}')
            with patch('tryon.repair.MODEL_DIR', root), patch('tryon.repair.REPAIR_PYTHON', root/'manifest.json'), \
                 patch('tryon.repair.OUTPUTS', root/'outputs'), patch('tryon.repair.subprocess.run', side_effect=worker):
                png, record, metadata = run_repair(*prepare_repair(self.editor, 'fabric', .75, 42))
            self.assertEqual(metadata['status'], 'needs_review')
            self.assertEqual(metadata['outside_mask_max_difference'], 0)
            self.assertTrue(Path(record).is_file())
            self.assertTrue((Path(png).parent/'before.png').is_file())
            self.assertTrue((Path(png).parent/'mask.png').is_file())
            with Image.open(png) as result:
                self.assertEqual(result.getpixel((25, 35)), (255, 0, 0))
                self.assertEqual(result.getpixel((0, 0)), (0, 0, 255))

    def test_timeout_leaves_no_published_candidate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root/'manifest.json').write_text('{}')
            with patch('tryon.repair.MODEL_DIR', root), patch('tryon.repair.REPAIR_PYTHON', root/'manifest.json'), \
                 patch('tryon.repair.OUTPUTS', root/'outputs'), \
                 patch('tryon.repair.subprocess.run', side_effect=subprocess.TimeoutExpired('worker', 300)):
                with self.assertRaisesRegex(RuntimeError, '5 分鐘'):
                    run_repair(*prepare_repair(self.editor, 'fabric', .75, 42))
            self.assertEqual(list((root/'outputs/repairs').iterdir()), [])

    def test_missing_install_is_explicit(self):
        with tempfile.TemporaryDirectory() as directory, patch('tryon.repair.MODEL_DIR', Path(directory)):
            with self.assertRaisesRegex(RuntimeError, 'setup_repair.sh'):
                run_repair(*prepare_repair(self.editor, 'fabric', .75, 42))

    @unittest.skipUnless(importlib.util.find_spec('cv2'), 'OpenCV is in the full environment, not UI-only')
    def test_texture_fill_removes_small_blemish_without_model_or_subprocess(self):
        damaged = self.base.copy()
        damaged.paste('magenta', (25, 35, 35, 50))
        editor = {**self.editor, 'background': damaged}
        with tempfile.TemporaryDirectory() as directory, patch('tryon.repair.OUTPUTS', Path(directory)), \
             patch('tryon.repair.subprocess.run') as worker:
            png, _, data = run_repair(*prepare_repair(editor, '', .99, 0, method='texture'))
            worker.assert_not_called()
            self.assertIsNone(data['model'])
            with Image.open(png) as result:
                self.assertLess(result.getpixel((30, 40))[0], 10)
                self.assertGreater(result.getpixel((30, 40))[2], 245)
                self.assertEqual(result.getpixel((0, 0)), (0, 0, 255))
