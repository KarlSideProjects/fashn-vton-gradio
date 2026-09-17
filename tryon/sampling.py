"""FASHN sampler with CPU FP32 initial noise for the GPU execution mode.

Adapted from FASHN AI (Copyright 2025), Apache-2.0; see licenses/Apache-2.0.txt and NOTICE.
Local modifications: see LICENSE and LICENSING.md; upstream rights are unchanged.
Source: fashn-vton-1.5, commit 7c0f10af3f91ad4048fe9729c470a13ef905d25a.
Only initial-noise placement/dtype differs; the Euler/CFG equations are unchanged.
This local override does not edit site-packages or globally patch torch.randn.
"""
import torch
from tqdm.auto import tqdm

from fashn_vton import TryOnPipeline
from fashn_vton.utils import get_rf_schedule, tensor_to_pil


class CpuNoisePipeline(TryOnPipeline):
    @torch.inference_mode()
    def _sample(
        self, *, ca_images, garment_images, person_poses, garment_poses,
        garment_categories, num_timesteps=30, time_shift_mu=1.5,
        guidance_scale=1.5, skip_cfg_last_n_steps=1, use_tqdm=True,
    ):
        device, dtype = ca_images.device, ca_images.dtype
        batch_size = ca_images.shape[0]
        c, h, w = self.tryon_model.channels_in, *self.tryon_model.input_shape
        # __call__ seeds the CPU RNG too. Cast only after generating FP32 noise.
        images = torch.randn((batch_size, c, h, w), dtype=torch.float32, device='cpu').to(device, dtype=dtype)
        timesteps = get_rf_schedule(num_steps=num_timesteps, mu=time_shift_mu)
        model_kwargs = {
            'person_poses': person_poses, 'garment_poses': garment_poses,
            'ca_images': ca_images, 'garment_images': garment_images,
            'garment_categories': garment_categories,
        }
        for step_idx, (t_curr, t_prev) in enumerate(tqdm(
            zip(timesteps[:-1], timesteps[1:]), desc='Sampling',
            total=len(timesteps) - 1, disable=not use_tqdm,
        )):
            dt = t_prev - t_curr
            t_vec = torch.full((batch_size,), t_curr, dtype=dtype, device=device)
            pred = self.tryon_model.forward_for_cfg(images, t_vec, **model_kwargs)
            v_c, v_u = pred['v_c'], pred['v_u']
            if skip_cfg_last_n_steps > 0 and step_idx >= num_timesteps - skip_cfg_last_n_steps:
                v_guided = v_c
            else:
                v_guided = v_u + guidance_scale * (v_c - v_u)
            images = images + dt * v_guided
        images = images.to(dtype=torch.float).clamp_(-1.0, 1.0)
        return [tensor_to_pil(img, unnormalize=True) for img in images]
