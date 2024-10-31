# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import random

import torch

from model import monotonic_align
from model.base import BaseModule
from model.text_encoder import TextEncoder
from model.diffusion import Diffusion
from model.shifter_model import build_shifter
from model.utils import (
    sequence_mask,
    generate_path,
    duration_loss,
    fix_len_compatibility,
    causal_mask,
    segment_sequence_to_batch,
)


class GradTTS(BaseModule):
    def __init__(
        self,
        n_vocab,
        n_spks,
        spk_emb_dim,
        n_enc_channels,
        filter_channels,
        filter_channels_dp,
        n_heads,
        n_enc_layers,
        enc_kernel,
        enc_dropout,
        window_size,
        n_feats,
        tgt_seq_len,
        dec_dim,
        beta_min,
        beta_max,
        pe_scale,
        device
    ):
        super(GradTTS, self).__init__()
        self.n_vocab = n_vocab
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_enc_channels = n_enc_channels
        self.filter_channels = filter_channels
        self.filter_channels_dp = filter_channels_dp
        self.n_heads = n_heads
        self.n_enc_layers = n_enc_layers
        self.enc_kernel = enc_kernel
        self.enc_dropout = enc_dropout
        self.window_size = window_size
        self.n_feats = n_feats
        self.tgt_seq_len = tgt_seq_len
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        self.device = device

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.encoder = TextEncoder(
            n_vocab,
            n_feats,
            n_enc_channels,
            filter_channels,
            filter_channels_dp,
            n_heads,
            n_enc_layers,
            enc_kernel,
            enc_dropout,
            window_size,
        )
        self.shifter = build_shifter(n_feats, tgt_seq_len=tgt_seq_len)
        self.decoder = Diffusion(
            n_feats, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale
        )

    @torch.no_grad()
    def forward(
        self,
        x,
        x_lengths,
        n_timesteps,
        temperature=1.0,
        stoc=False,
        spk=None,
        length_scale=1.0,
        out_size=172,
    ):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)

        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale  # (1, 1, 55)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(
            y_max_length
        )  # so that y_max_length is multiple of (4)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = (
            sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        )  # (1, 1, 200)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(
            2
        )  # (1, 1, 55, 1) * (1, 1, 1, 200) = (1, 1, 55, 200)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(
            1
        )  # (1, 1, 55, 200)

        # Align encoded text and get mu_y
        m = torch.matmul(
            attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)
        )  # (1, 200, 55) (1, 55, 80) align \tilde{\mu} to \mu

        encoder_output = m[:, :y_max_length, :] # (1, 197, 80)

        decoder_inputs = torch.full((1, 1, self.n_feats), -1).type(m.dtype).to(x.device) ##TODO: check mu_y.dtype
        
        while decoder_inputs.size(1) <= y_max_length_:
            # build mask for target and calculate output
            decoder_mask = torch.triu(torch.ones((1, decoder_inputs.size(1), decoder_inputs.size(1))), diagonal=1).type(torch.int).to(x.device)
            out = self.shifter.decode(m, y_mask.unsqueeze(1), decoder_inputs, decoder_mask.unsqueeze(1), None)

            # project next token
            predicted_next_frame = self.shifter.project(out[:, -1])
            decoder_inputs = torch.cat([decoder_inputs, predicted_next_frame.unsqueeze(1)], dim=1)
        
        mu_y = decoder_inputs[:, 1:, :].transpose(1, 2)
        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(
            z, y_mask, mu_y, n_timesteps, stoc, spk
        )  # (1, 80, 200)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return encoder_output.transpose(1, 2), mu_y, decoder_outputs, attn[:, :, :y_max_length]

    def compute_loss(self, x, x_lengths, y, y_lengths, spk=None, out_size=None):
        """
        Computes 3 losses:
            1. duration loss: loss between predicted token durations and those extracted by Monotinic Alignment Search (MAS).
            2. prior loss: loss between mel-spectrogram and encoder outputs.
            3. diffusion loss: loss between gaussian noise and its reconstruction by diffusion-based decoder.

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            y (torch.Tensor): batch of corresponding mel-spectrograms.
            y_lengths (torch.Tensor): lengths of mel-spectrograms in batch.
            out_size (int, optional): length (in mel's sampling rate) of segment to cut, on which decoder will be trained.
                Should be divisible by 2^{num of UNet downsamplings}. Needed to increase batch size.
        """
        x, x_lengths, y, y_lengths = self.relocate_input([x, x_lengths, y, y_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(
            2
        )  # (16, 1, 265, 1) * (16, 1, 1, 812) = (16, 1, 265, 812)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y**2)  # (16, 265, 812)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(
                log_prior, attn_mask.squeeze(1)
            )  # (16, 265, 812)
            attn = attn.detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            # middle = y_lengths // 2
            offset_ranges = list(
                zip([0] * y_lengths.shape[0], max_offset.cpu().numpy())
            )
            out_offset = torch.LongTensor(
                [
                    torch.tensor(random.choice(range(start, end)) if end > start else 0)
                    for start, end in offset_ranges
                ]
            ).to(y_lengths)

            attn_cut = torch.zeros(
                attn.shape[0],
                attn.shape[1],
                out_size,
                dtype=attn.dtype,
                device=attn.device,
            )  # (16, 265, 172)
            y_cut = torch.zeros(
                y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device
            )  # (16, 80, 172)
            y_cut_lengths = []
            # y: (16, 80, 812) out_offset: (16)

            # if length < out_size, sample the whole segment
            for i, (y_, out_offset_) in enumerate(zip(y, out_offset)):
                y_cut_length = min(out_size, y_lengths[i])
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        m = torch.matmul(
            attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)
        )  # (16, 172, 265), (16, 265, 80) = (16, 172, 80)
        
        sos_vector = torch.full((m.shape[0], 1, m.shape[2]), -1).to(self.device) ##TODO: effective way to check device 
        decoder_input = torch.cat((sos_vector, y.transpose(1, 2)[:, :-1, :]), 1)
        
        sos_mask = torch.full((y_mask.shape[0], y_mask.shape[1], 1), 1).to(self.device)
        y_mask_ = torch.cat((sos_mask, y_mask[:, :, :-1]), 2)
        
        tgt_mask = torch.cat([y_mask[i].int() & causal_mask(out_size).to(self.device) for i in range(y_mask_.shape[0])], 0)
        
        mu_y = self.shifter.decode(m, y_mask_.unsqueeze(1), decoder_input, tgt_mask.unsqueeze(1), None)
        mu_y = self.shifter.project(mu_y).transpose(1, 2) # (16, 80, 172)

        # Compute loss of score-based decoder
        diff_loss, xt, noise_estimation, noise_reference = self.decoder.compute_loss(y, y_mask, mu_y, spk)

        # Compute loss between aligned encoder outputs and mel-spectrogram

        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

        return dur_loss, prior_loss, diff_loss, noise_estimation, noise_reference
