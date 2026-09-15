"""Small numeric sampler checks; no weights or generated substitute photos."""
import importlib.util
import unittest
from types import SimpleNamespace


@unittest.skipUnless(importlib.util.find_spec('torch') and importlib.util.find_spec('fashn_vton'),
                     'Numeric checks need the model environment, not just UI dependencies')
class SamplingTests(unittest.TestCase):
    def sample(self, cls, device, dtype, seed):
        import torch
        seen = []
        def forward(images, t, **kwargs):
            seen.append(images.clone())
            return {'v_c': images * .1, 'v_u': images * .05}
        sampler = cls.__new__(cls)
        sampler.tryon_model = SimpleNamespace(channels_in=3, input_shape=(4, 4), forward_for_cfg=forward)
        values = torch.zeros((1, 3, 4, 4), device=device, dtype=dtype)
        torch.manual_seed(seed)
        result = sampler._sample(ca_images=values, garment_images=values, person_poses=values,
            garment_poses=values, garment_categories=torch.ones(1, device=device),
            num_timesteps=3, use_tqdm=False)
        return result[0], seen[0]

    def test_cpu_math_matches_upstream_for_multiple_seeds(self):
        import torch
        from fashn_vton import TryOnPipeline
        from tryon.sampling import CpuNoisePipeline
        randn = torch.randn
        for seed in (0, 42, 57):
            base, noise = self.sample(TryOnPipeline, 'cpu', torch.float32, seed)
            local, local_noise = self.sample(CpuNoisePipeline, 'cpu', torch.float32, seed)
            self.assertTrue(torch.equal(noise, local_noise))
            self.assertEqual(base.tobytes(), local.tobytes())
        self.assertIs(torch.randn, randn)

    def test_gpu_initial_noise_is_cpu_sample_cast_to_gpu_dtype(self):
        import torch
        from tryon.sampling import CpuNoisePipeline
        if not torch.cuda.is_available():
            self.skipTest('No CUDA hardware')
        dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32
        for seed in (42, 57):
            _, cpu = self.sample(CpuNoisePipeline, 'cpu', torch.float32, seed)
            _, gpu = self.sample(CpuNoisePipeline, 'cuda', dtype, seed)
            self.assertTrue(torch.equal(cpu.to('cuda', dtype=dtype), gpu))
