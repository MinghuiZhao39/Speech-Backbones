""" from https://github.com/jaywalnut310/glow-tts """

import torch


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1) #(1, 55) (1, 1)


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1


def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape


def generate_path(duration, mask):
    device = duration.device

    b, t_x, t_y = mask.shape #(1, 55, 200)
    cum_duration = torch.cumsum(duration, 1) #cummulative duration 
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device) #(1, 55)

    cum_duration_flat = cum_duration.view(b * t_x) #(55)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype) #(55, 200)
    path = path.view(b, t_x, t_y) #(1, 55, 200)
    path = path - torch.nn.functional.pad(path, convert_pad_shape([[0, 0], 
                                          [1, 0], [0, 0]]))[:, :-1] #difference with its offset by 1
    path = path * mask
    return path


def duration_loss(logw, logw_, lengths):
    loss = torch.sum((logw - logw_)**2) / torch.sum(lengths)
    return loss
