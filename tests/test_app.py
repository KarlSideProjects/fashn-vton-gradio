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
        self.assertEqual(types.count('image'), 4)
        self.assertEqual(types.count('imageeditor'), 1)
        repair_event = next(d for d in ui.config['dependencies'] if d.get('api_name') == 'local_repair')
        self.assertEqual(len(repair_event['inputs']), 6)

    def test_generated_result_automatically_loads_repair_editor(self):
        ui = app.build_app()
        generation = next(d for d in ui.config['dependencies'] if d.get('api_name') == 'try_on')
        result_id = generation['outputs'][0]
        editor_id = next(c['id'] for c in ui.config['components'] if c['type'] == 'imageeditor')
        event = next(d for d in ui.config['dependencies'] if (result_id, 'change') in d['targets'])
        self.assertEqual(event['inputs'], [result_id])
        self.assertEqual(event['outputs'], [editor_id])
        callback = ui.fns[event['id']].fn
        for color in ('red', 'blue'):
            photo = Image.new('RGB', (100, 100), color)
            editor = callback(photo)
            self.assertIs(editor['background'], photo)
            self.assertIs(editor['composite'], photo)
            self.assertEqual(editor['layers'], [])
        # Progress/failure clears the result; keep any existing repair work intact.
        self.assertEqual(callback(None), app.gr.skip())

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

    def test_repair_failure_keeps_input_and_clears_previous_result(self):
        editor = {'background': Image.new('RGBA', (100, 100)), 'layers': []}
        with self.assertLogs('app', level='ERROR'):
            updates = list(app.repair(editor, 'blue fabric', 0.75, 42))
        self.assertTrue(all(update[:3] == (None, None, None) for update in updates))
        self.assertIn('尚未塗選', updates[-1][3])
        self.assertEqual(editor['background'].size, (100, 100))


if __name__ == '__main__':
    unittest.main()
