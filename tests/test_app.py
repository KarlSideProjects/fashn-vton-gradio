import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import app


class AppTests(unittest.TestCase):
    def test_builds_real_gradio_interface(self):
        ui = app.build_app()
        types = [component['type'] for component in ui.config['components']]
        self.assertEqual(types.count('image'), 3)

    def test_failure_clears_old_image_and_downloads(self):
        with self.assertLogs('app', level='ERROR'):
            updates = list(app.generate(None, None, 'tops', 'model', 30, 42))
        self.assertTrue(all(update[:3] == (None, None, None) for update in updates))
        self.assertIn('請上傳', updates[-1][3])

    def test_real_engine_reports_missing_weights_without_fake_result(self):
        photo = Image.new('RGB', (100, 100))
        with tempfile.TemporaryDirectory() as empty:
            with patch('tryon.engine.WEIGHTS', Path(empty)), self.assertLogs('app', level='ERROR'):
                updates = list(app.generate(photo, photo, 'tops', 'model', 30, 42))
        self.assertIn('缺少模型權重', updates[-1][3])
        self.assertIsNone(updates[-1][0])


if __name__ == '__main__':
    unittest.main()
