import json
import tempfile
import unittest
from pathlib import Path

from PIL import Image

from tryon.core import prepare_image, validate_options, save_result


class CoreTests(unittest.TestCase):
    def test_missing_image_is_rejected(self):
        with self.assertRaisesRegex(ValueError, '圖片'):
            prepare_image(None)

    def test_transparent_image_uses_white_background(self):
        result = prepare_image(Image.new('RGBA', (100, 100), (0, 0, 0, 0)))
        self.assertEqual(result.mode, 'RGB')
        self.assertEqual(result.getpixel((0, 0)), (255, 255, 255))

    def test_exif_rotation_is_applied(self):
        photo = Image.new('RGB', (100, 150))
        photo.getexif()[274] = 6
        self.assertEqual(prepare_image(photo).size, (150, 100))

    def test_parameters_cannot_exceed_supported_budget(self):
        for change in [dict(category='shoes'), dict(seed=-1), dict(seed=2**32),
                       dict(steps=1000), dict(steps=20.5), dict(photo_type='auto')]:
            args = dict(category='tops', photo_type='model', steps=30, seed=42)
            args.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_options(**args)

    def test_valid_parameters(self):
        self.assertEqual(validate_options('tops', 'model', 30, 42)['num_samples'], 1)

    def test_each_result_has_unique_png_and_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            photo = Image.new('RGB', (100, 100))
            first = save_result(Path(directory), photo, {'seed': 42})
            second = save_result(Path(directory), photo, {'seed': 42})
            self.assertNotEqual(first, second)
            with Image.open(first[0]) as saved:
                self.assertEqual(saved.size, photo.size)
            self.assertEqual(json.loads(first[1].read_text())['seed'], 42)


if __name__ == '__main__':
    unittest.main()
