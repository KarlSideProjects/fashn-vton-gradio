"""Checks for the imported single-pass Space flow; no model downloads."""
import sys
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image

import app


class AppTests(unittest.TestCase):
    def test_ui_has_only_try_on_and_no_repair(self):
        types = [c['type'] for c in app.demo.config['components']]
        self.assertEqual(types.count('image'), 3)
        self.assertNotIn('imageeditor', types)
        events = [d for d in app.demo.config['dependencies'] if d.get('api_name') == 'try_on']
        self.assertEqual(len(events), 1)
        self.assertEqual(len(events[0]['inputs']), 9)
        self.assertEqual(len(events[0]['outputs']), 2)
        self.assertIs(app.demo.fns[events[0]['id']].fn, app.try_on)
        self.assertEqual(app.demo._queue.default_concurrency_limit, 1)

    def test_returns_exact_pipeline_image_once_with_space_parameters(self):
        person = Image.new('RGB', (100, 100), 'red')
        garment = np.zeros((100, 100, 3), dtype=np.uint8)
        expected = Image.new('RGB', (100, 100), 'blue')
        pipeline = Mock(return_value=SimpleNamespace(images=[expected]))
        with patch('app.get_pipeline', return_value=pipeline):
            actual, status = app.try_on(person, garment, 'tops', 'model', 30, 1.5, 42, True)
        self.assertIs(actual, expected)  # No copies, composites, repair, or candidate selection.
        self.assertIn('Done', status)
        pipeline.assert_called_once()
        kwargs = pipeline.call_args.kwargs
        self.assertEqual(kwargs.pop('person_image').tobytes(), person.tobytes())
        self.assertEqual(kwargs.pop('garment_image').mode, 'RGB')
        self.assertEqual(kwargs, dict(category='tops', garment_photo_type='model', num_samples=1,
                         num_timesteps=30, guidance_scale=1.5, seed=42, segmentation_free=True))

    def test_cpu_loader_is_cached_and_matches_space(self):
        factory = Mock()
        with (patch('app._pipeline', None), patch('app._pipeline_device', None), patch('app.os.path.isfile', return_value=True),
              patch.dict(sys.modules, {'fashn_vton': SimpleNamespace(TryOnPipeline=factory)})):
            first = app.get_pipeline()
            self.assertIs(app.get_pipeline(), first)
        factory.assert_called_once_with(weights_dir=app.WEIGHTS_DIR, device='cpu')

    def test_missing_images_and_invalid_budgets_do_not_load_model(self):
        photo = Image.new('RGB', (100, 100))
        base = dict(person_image=photo, garment_image=photo, category='tops',
                    garment_photo_type='model', num_timesteps=30, guidance_scale=1.5,
                    seed=42, segmentation_free=True)
        cases = [dict(person_image=None), dict(garment_image=None), dict(category='shoes'),
                 dict(num_timesteps=1000), dict(num_timesteps=30.5), dict(guidance_scale=float('nan')),
                 dict(seed=float('inf')), dict(segmentation_free=1),
                 dict(person_image=Image.new('RGB', (10, 10)))]
        with patch('app.get_pipeline') as loader:
            for case in cases:
                with self.subTest(case=case), self.assertRaises(app.gr.Error):
                    app.try_on(**(base | case))
        loader.assert_not_called()

    def test_seed_fallback_matches_space(self):
        photo = Image.new('RGB', (100, 100))
        for seed in (None, -1):
            pipeline = Mock(return_value=SimpleNamespace(images=[photo]))
            with patch('app.get_pipeline', return_value=pipeline):
                app.try_on(photo, photo, 'tops', 'model', 30, 1.5, seed, True)
            self.assertEqual(pipeline.call_args.kwargs['seed'], 42)

    def test_device_switch_releases_previous_pipeline_and_does_not_cache_both(self):
        cpu_factory, gpu_factory = Mock(), Mock()
        cuda = SimpleNamespace(is_available=lambda: True, empty_cache=Mock())
        with (patch('app._pipeline', None), patch('app._pipeline_device', None),
              patch('app.os.path.isfile', return_value=True), patch('app.gc.collect') as collect,
              patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=cuda),
                  'onnxruntime': SimpleNamespace(),
                  'fashn_vton': SimpleNamespace(TryOnPipeline=cpu_factory),
                  'tryon.sampling': SimpleNamespace(CpuNoisePipeline=gpu_factory)})):
            first = app.get_pipeline('cpu')
            self.assertIs(app.get_pipeline('cpu'), first)
            self.assertIs(app.get_pipeline('cuda'), gpu_factory.return_value)
            self.assertIs(app.get_pipeline('cpu'), cpu_factory.return_value)
            self.assertEqual(app._pipeline_device, 'cpu')
        self.assertEqual(cpu_factory.call_count, 2)
        gpu_factory.assert_called_once_with(weights_dir=app.WEIGHTS_DIR, device='cuda')
        self.assertEqual(collect.call_count, 2)
        cuda.empty_cache.assert_called_once()

    def test_unavailable_gpu_never_silently_falls_back(self):
        existing = object()
        with (patch('app._pipeline', existing), patch('app._pipeline_device', 'cpu'),
              patch.dict(sys.modules, {'torch': SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: False))})):
            with self.assertRaisesRegex(RuntimeError, 'GPU 不可用'):
                app.get_pipeline('cuda')
            self.assertIs(app._pipeline, existing)

    def test_switching_request_waits_for_current_inference(self):
        photo = Image.new('RGB', (100, 100))
        entered, release = threading.Event(), threading.Event()
        results = []
        def generate(**kwargs):
            entered.set()
            self.assertTrue(release.wait(2))
            return SimpleNamespace(images=[photo])
        with patch('app.get_pipeline', return_value=generate) as loader:
            one = threading.Thread(target=lambda: results.append(app.try_on(photo, photo, 'tops', 'model', 30, 1.5, 42, True, 'cpu')))
            two = threading.Thread(target=lambda: results.append(app.try_on(photo, photo, 'tops', 'model', 30, 1.5, 42, True, 'cuda')))
            one.start()
            self.assertTrue(entered.wait(1))
            two.start()
            self.assertEqual(loader.call_count, 1)
            release.set()
            one.join(3)
            two.join(3)
        self.assertEqual([call.args[0] for call in loader.call_args_list], ['cpu', 'cuda'])
        self.assertEqual(len(results), 2)
        self.assertTrue(all(r[0] is photo for r in results))

    def test_load_or_inference_error_returns_no_stale_image_and_no_retry(self):
        photo = Image.new('RGB', (100, 100))
        with patch('app.get_pipeline', side_effect=RuntimeError('missing weights')) as loader:
            result, status = app.try_on(photo, photo, 'tops', 'model', 30, 1.5, 42, True)
        self.assertIsNone(result)
        self.assertIn('missing weights', status)
        loader.assert_called_once()
        pipeline = Mock(side_effect=RuntimeError('failed'))
        with patch('app.get_pipeline', return_value=pipeline):
            result, status = app.try_on(photo, photo, 'tops', 'model', 30, 1.5, 42, True)
        self.assertIsNone(result)
        self.assertIn('failed', status)
        pipeline.assert_called_once()


if __name__ == '__main__':
    unittest.main()
