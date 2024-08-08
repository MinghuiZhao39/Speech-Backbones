import math
import torch
from einops import rearrange

from model.base import BaseModule


class Mish(BaseModule):
    def forward(self, x):
        return x * torch.tanh(torch.nn.functional.softplus(x))


class Upsample(BaseModule):
    def __init__(self, dim):
        super(Upsample, self).__init__()
        self.conv = torch.nn.ConvTranspose1d(dim, dim, 4, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Downsample(BaseModule):
    def __init__(self, dim):
        super(Downsample, self).__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class Rezero(BaseModule):
    def __init__(self, fn):
        super(Rezero, self).__init__()
        self.fn = fn
        self.g = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return self.fn(x) * self.g


class Block(BaseModule):
    def __init__(self, dim, dim_out, groups=8):
        super(Block, self).__init__()
        self.block = torch.nn.Sequential(torch.nn.Conv1d(dim, dim_out, 3, 
                                         padding=1), torch.nn.GroupNorm(
                                         groups, dim_out), Mish())

    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask


class ResnetBlock(BaseModule):
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super(ResnetBlock, self).__init__()
        self.mlp = torch.nn.Sequential(Mish(), torch.nn.Linear(time_emb_dim, 
                                                               dim_out))

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        if dim != dim_out:
            self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)
        else:
            self.res_conv = torch.nn.Identity()

    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask) # (1024, 64, 80)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output


class LinearAttention(BaseModule):
    def __init__(self, dim, heads=4, dim_head=32):
        super(LinearAttention, self).__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = torch.nn.Conv1d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = torch.nn.Conv1d(hidden_dim, dim, 1)            

    def forward(self, x):
        b, c, l = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) l -> qkv b heads c l', 
                            heads = self.heads, qkv=3)            
        k = k.softmax(dim=-1)
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c l -> b (heads c) l', 
                        heads=self.heads, l=l)
        return self.to_out(out)


class Residual(BaseModule):
    def __init__(self, fn):
        super(Residual, self).__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        output = self.fn(x, *args, **kwargs) + x
        return output


class SinusoidalPosEmb(BaseModule):
    def __init__(self, dim):
        super(SinusoidalPosEmb, self).__init__()
        self.dim = dim

    def forward(self, x, scale=1000):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class GradLogPEstimator1d(BaseModule):
    def __init__(self, dim, dim_mults=(1, 2, 4), groups=8,
                 n_spks=None, spk_emb_dim=64, n_feats=80, pe_scale=1000):
        super(GradLogPEstimator1d, self).__init__()
        self.dim = dim
        self.dim_mults = dim_mults
        self.groups = groups
        self.n_spks = n_spks if not isinstance(n_spks, type(None)) else 1
        self.spk_emb_dim = spk_emb_dim
        self.pe_scale = pe_scale
        
        if n_spks > 1:
            self.spk_mlp = torch.nn.Sequential(torch.nn.Linear(spk_emb_dim, spk_emb_dim * 4), Mish(),
                                               torch.nn.Linear(spk_emb_dim * 4, n_feats))
        self.time_pos_emb = SinusoidalPosEmb(dim)
        self.mlp = torch.nn.Sequential(torch.nn.Linear(dim, dim * 4), Mish(),
                                       torch.nn.Linear(dim * 4, dim))

        dims = [2 + (1 if n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        self.downs = torch.nn.ModuleList([])
        self.ups = torch.nn.ModuleList([])
        num_resolutions = len(in_out) # [(2, 64), (64, 128), (128, 256)]

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                       ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                       Residual(Rezero(LinearAttention(dim_out))),
                       Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        mid_dim = dims[-1]
        self.mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.ups.append(torch.nn.ModuleList([
                     ResnetBlock(dim_out * 2, dim_in, time_emb_dim=dim),
                     ResnetBlock(dim_in, dim_in, time_emb_dim=dim),
                     Residual(Rezero(LinearAttention(dim_in))),
                     Upsample(dim_in)]))
        self.final_block = Block(dim, dim)
        self.final_conv = torch.nn.Conv1d(dim, 1, 1)

    def forward(self, x, mask, mu, t, spk=None):
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)
        
        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        if self.n_spks < 2:
            x = torch.stack([mu, x], 1).squeeze(2) #(16, 80, 172) (16, 80, 172) = (16, 2, 80, 172)
        else:
            s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.stack([mu, x, s], 1)
        # mask = mask.unsqueeze(1) #(16, 1, 1, 172)

        hiddens = []
        masks = [mask]
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t) #(16, 64, 80) (16, 64, 80, 172)| (16, 128, 40, 86) | (16, 256, 20, 43)
            x = resnet2(x, mask_down, t) #(16, 64, 80) | (16, 128, 40, 86) | (16, 256, 20, 43)
            x = attn(x) #(16, 64, 80) | (16, 128, 40, 86) | (16, 256, 20, 43)
            hiddens.append(x)
            x = downsample(x * mask_down) #(16, 64, 40) | (16, 128, 20, 43) | (16, 256, 20, 43)
            # masks.append(mask_down[:, :, :, ::2])
            masks.append(mask_down)

        masks = masks[:-1]
        mask_mid = masks[-1] # (16, 1, 1, 43)
        x = self.mid_block1(x, mask_mid, t) #(16, 256, 20, 43)
        x = self.mid_attn(x) # (16, 256, 20, 43)
        x = self.mid_block2(x, mask_mid, t) # (16, 256, 20, 43)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask) #(16, 1, 80, 172)
        output = self.final_conv(x * mask)

        return (output * mask) #(16, 1, 80, 172) (16, 1, 1, 172) = (16, 1, 80, 172) (16, 80, 172)


def get_noise(t, beta_init, beta_term, cumulative=False):
    if cumulative:
        noise = beta_init*t + 0.5*(beta_term - beta_init)*(t**2)
    else:
        noise = beta_init + (beta_term - beta_init)*t
    return noise


class FrameLevelDiffusion(BaseModule):
    def __init__(self, n_feats, dim,
                 n_spks=1, spk_emb_dim=64,
                 beta_min=0.05, beta_max=20, pe_scale=1000):
        super(FrameLevelDiffusion, self).__init__()
        self.n_feats = n_feats
        self.dim = dim # 64
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale
        
        self.estimator = GradLogPEstimator1d(dim, n_spks=n_spks,
                                             spk_emb_dim=spk_emb_dim,
                                             pe_scale=pe_scale)

    def forward_diffusion(self, x0, mask, mu, t):
        time = t.unsqueeze(-1).unsqueeze(-1) # (16, 1, 1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        mean = x0*torch.exp(-0.5*cum_noise) + mu*(1.0 - torch.exp(-0.5*cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device, 
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask, cum_noise 

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, n_timesteps, stoc=False, spk=None): 
        # mask (1, 1, 200) z (1, 80, 200) mu (1, 80, 200)
        h = 1.0 / n_timesteps
        xt = z * mask
        t = torch.tensor([1.0 - (i + 0.5) * h for i in range(n_timesteps)])
        time = t.unsqueeze(-1).unsqueeze(-1)
        noise_t = get_noise(time, self.beta_min, self.beta_max, cumulative=False)
        
        ## original diffusion calculated frame by frame
        # for i in range(n_timesteps):
        #     noise_estimate = self.estimator(xt, mask, mu, t[i:i+1], spk)
        #     generated_frames = torch.zeros_like(mu)
        #     for frame_num in range(mu.shape[-1]):
        #         current_frame = xt[:, :, frame_num]
        #         dxt = 0.5 * (mu[:, :, frame_num] - current_frame - noise_estimate[:, :, frame_num])
        #         dxt = dxt * noise_t[i] * h
        #         current_mask = mask[:, :, frame_num]
        #         current_frame = (current_frame - dxt) * current_mask 
        #         generated_frames[:, :, frame_num] = current_frame
        #     xt = generated_frames
        
        # frame level diffusion with intermediate steps
        generated_frames = torch.zeros_like(mu)
        for frame_num in range(mu.shape[-1]):

            current_frame = xt[:, :, frame_num]
            current_mu = mu[:, :, frame_num]
            current_mask = mask[:, :, frame_num]

            for i in range(n_timesteps):
                dxt = 0.5 * (current_mu - current_frame - self.estimator(current_frame, current_mask, current_mu, t[i:i+1], spk))
                dxt = dxt * noise_t[i] * h
                current_frame = (current_frame - dxt) * current_mask
            
            generated_frames[:, :, frame_num] = current_frame
        return generated_frames
        

    @torch.no_grad()
    def forward(self, z, mask, mu, n_timesteps, stoc=False, spk=None):
        return self.reverse_diffusion(z, mask, mu, n_timesteps, stoc, spk)

    def loss_t(self, x0, mask, mu, t, spk=None):
        xt, z, cum_noise  = self.forward_diffusion(x0, mask, mu, t) #(16, 80, 172)
        noise_estimation = self.estimator(xt, mask, mu, t, spk) #(16, 80, 172)
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise)) # \sqrt(lambda)s_\theta
        loss = torch.sum((noise_estimation + z)**2) / (torch.sum(mask)*self.n_feats)
        return loss, xt

    def compute_loss(self, x0, mask, mu, spk=None, offset=1e-5):
        # x0 (16, 80, 64) mask (16, 1, 64) mu (16, 80, 64)
        x0_ = x0.transpose(1, 2).reshape(-1, 80).unsqueeze(1)
        mu_ = mu.transpose(1, 2).reshape(-1, 80).unsqueeze(1)
        mask_ = mask.transpose(1, 2).reshape(-1, 1).unsqueeze(1)
        t = torch.rand(x0_.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0_, mask_, mu_, t, spk)
