# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import torch
from einops import rearrange

from model.base import BaseModule
from model.model_layers import (
    SinusoidalPosEmb,
    Mish,
    Residual,
    ResnetBlock,
    LinearAttention,
    Rezero,
    Downsample,
    Upsample,
    Block,
)


class GradLogPEstimator2d(BaseModule):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4),
        groups=8,
        n_spks=None,
        spk_emb_dim=64,
        n_feats=80,
        pe_scale=1000,
    ):
        super(GradLogPEstimator2d, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.n_spks = n_spks if not isinstance(n_spks, type(None)) else 1
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale

        if n_spks > 1:
            self.spk_mlp = torch.nn.Sequential(
                torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4),
                Mish(),
                torch.nn.Linear(spk_emb_dim * 4, n_feats),
            )
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = torch.nn.Sequential(
            torch.nn.Linear(dim, dim * 4), Mish(), torch.nn.Linear(dim * 4, dim)
        )

        dims = [2 + (1 if n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                        Residual(Rezero(LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else torch.nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(
                torch.nn.ModuleList(
                    [
                        ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                        ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                        Residual(Rezero(LinearAttention(dim_in))),
                        Upsample(dim_in),
                    ]
                )
            )
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv2d(dim, 1, 1)

    def forward(self, x, mask, mu, t, spk=None):
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)

        t = self.time_pos_emb(t, scale=self.pe_scale)  # (16, 64)
        t = self.mlp(t)  # (16, 64)

        if self.n_spks < 2:
            x = torch.stack(
                [mu, x], 1
            )  # (16, 80, 172) (16, 80, 172) = (16, 2, 80, 172)
        else:
            s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.stack([mu, x, s], 1)
        mask = mask.unsqueeze(1)  # (16, 1, 1, 172)

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t) # (16, 64, 80, 172) (16, 128, 80, 86) (16, 256, 20, 43)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down) # (16, 64, 40, 86) (16, 128, 20, 43) (16, 256, 20, 43)
            masks.append(mask_down[:, :, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1) # (16, 512, 20, 43) (16, 256, 40, 86)
            x = resnet1(x, mask_up, t) # (16, 128, 20, 43) (16, 64, 40, 86)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up) # (16, 128, 40, 86) (16, 64, 80, 172)

        x = self.final_block(x, mask) # (16, 64, 80, 172)
        output = self.final_conv(x * mask) # (16, 1, 80, 172)

        return (output * mask).squeeze(1)


def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        noise = beta_init * t + 0.5 * (beta_term - beta_init) * (t**2)
    else:
        noise = beta_init + (beta_term - beta_init) * t
    return noise


class Diffusion(BaseModule):
    def __init__(
        self,
        n_feats,
        dim,
        n_spks=1,
        spk_emb_dim=64,
        beta_min=0.05,
        beta_max=20,
        pe_scale=1000,
    ):
        super(Diffusion, self).__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        self.estimator = GradLogPEstimator2d(
            dim, n_spks=n_spks, spk_emb_dim=spk_emb_dim, pe_scale=pe_scale
        )

    def forward_diffusion(self, x0, mask, mu, t):
        time = t.unsqueeze(-1).unsqueeze(-1)  # (16, 1, 1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        mean = x0 * torch.exp(-0.5 * cum_noise) + mu * (
            1.0 - torch.exp(-0.5 * cum_noise)
        )
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max, cumulative=False)
            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (mu - xt) - self.estimator(xt, mask, mu, t, spk)
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(
                    z.shape, dtype=z.dtype, device=z.device, requires_grad=False
                )
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t, spk))
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def forward(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc, spk)

    def loss_t(self, x0, mask, mu, t, spk=None):
        xt, z = self.forward_diffusion(x0, mask, mu, t)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        noise_estimation = self.estimator(xt, mask, mu, t, spk)
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum((noise_estimation + z) ** 2) / (torch.sum(mask) * self.n_feats)
        return loss, xt

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        t = torch.rand(
            x0.shape[0], dtype=x0.dtype, device=x0.device, requires_grad=False
        )
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, mu, t, spk)
