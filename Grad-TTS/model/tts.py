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
from model.shifter_model import build_attention_aligner
from model.utils import sequence_mask, fix_len_compatibility, causal_mask, create_eos_labels


class GradTTS(BaseModule):
    def __init__(self, n_vocab, n_spks, spk_emb_dim, n_enc_channels, filter_channels, filter_channels_dp, 
                 n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size, 
                 n_feats, dec_dim, beta_min, beta_max, pe_scale, device):
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
        self.dec_dim = dec_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        self.device = device

        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.encoder = TextEncoder(n_vocab, n_feats, n_enc_channels, 
                                   filter_channels, filter_channels_dp, n_heads, 
                                   n_enc_layers, enc_kernel, enc_dropout, window_size)
        self.decoder = Diffusion(n_feats, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale)
        self.attention_align = build_attention_aligner(n_feats) 

    @torch.no_grad()
    def forward(self, x, x_lengths, n_timesteps, temperature=1.0, stoc=False, spk=None, length_scale=1.0):
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
        mu_x, x_mask = self.encoder(x, x_lengths, spk)

        # multihead attention and encoder
        mu_y, y_lengths = self.attention_align(mu_x)

        mu_y = mu_y.transpose(1, 2)

        y_max_length = int(y_lengths.max())
        y_max_length_ = fix_len_compatibility(y_max_length) # so that y_max_length is multiple of (4)

        # Using obtained durations `w` construct alignment map `attn`
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype) #(1, 1, 200)

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, mu_y, n_timesteps, stoc, spk) #(1, 80, 200)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return decoder_outputs,  mu_y[:, :, :y_max_length]

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
        labels = create_eos_labels(y, y_lengths)

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)
        
        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, x_mask = self.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)

        sos_vector = torch.full((mu_x.shape[0], 1, self.n_feats), -1).to(self.device) ##TODO: effective way to check device 
        decoder_input = torch.cat((sos_vector, y.transpose(1, 2)[:, :-1, :]), 1)
        
        sos_mask = torch.full((y_mask.shape[0], y_mask.shape[1], 1), 1).to(self.device)
        y_mask_ = torch.cat((sos_mask, y_mask[:, :, :-1]), 2)
        
        tgt_mask = torch.cat([y_mask_[i].int() & causal_mask(y_max_length).to(self.device) for i in range(y_mask_.shape[0])], 0)
        
        mu_y, eos_loss = self.attention_align.compute_encoder_output_n_eos_loss(mu_x, y_mask_, decoder_input, tgt_mask.unsqueeze(1), y_mask, labels)


        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)
            
            mu_y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device) #(16, 80, 172)
            labels_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device) #(16, 80, 172)
            mu_y_cut_lengths = []
            # y: (16, 80, 812) out_offset: (16)
            for i, (mu_y_, out_offset_, labels_) in enumerate(zip(mu_y, out_offset, labels)):
                mu_y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                mu_y_cut_lengths.append(mu_y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + mu_y_cut_length
                mu_y_cut[i, :, :mu_y_cut_length] = mu_y_[:, cut_lower:cut_upper]
                labels_cut[i, :, :mu_y_cut_length] = labels_[:, cut_lower:cut_upper]
            mu_y_cut_lengths = torch.LongTensor(mu_y_cut_lengths)
            mu_y_cut_mask = sequence_mask(mu_y_cut_lengths).unsqueeze(1).to(y_mask)
            
            mu_y = mu_y_cut
            mu_y_mask = mu_y_cut_mask

        # Compute loss of score-based decoder
        diff_loss, xt, noise_estimation, noise_ref = self.decoder.compute_loss(y, mu_y_mask, mu_y, spk)
        
        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        
        return eos_loss, prior_loss, diff_loss
