""" from https://github.com/jaywalnut310/glow-tts """

import torch


def sequence_mask(length, max_length=None):
    if max_length is None:
        max_length = length.max()
    x = torch.arange(int(max_length), dtype=length.dtype, device=length.device)
    return x.unsqueeze(0) < length.unsqueeze(1) #(1, 55) (1, 1)

def causal_mask(size):
    """
    Generates a causal mask for a sequence of a given size.

    The causal mask is an upper triangular matrix with ones above the diagonal
    and zeros on and below the diagonal. This mask is used to prevent the model
    from attending to future tokens in a sequence.

    Args:
        size (int): The size of the sequence for which the mask is generated.

    Returns:
        torch.Tensor: A boolean tensor of shape (1, size, size) where True indicates
                      positions that are allowed to be attended to and False indicates
                      positions that are masked.
    """
    mask = torch.triu(torch.ones((1, size, size)), diagonal=1).type(torch.int)
    return mask == 0


def fix_len_compatibility(length, num_downsamplings_in_unet=2):
    while True:
        if length % (2**num_downsamplings_in_unet) == 0:
            return length
        length += 1

def segment_sequence_to_batch(sequence, batch_size, dim):
    if sequence.size(dim) % batch_size != 0:
        raise ValueError("sequence length (dim = 1) must be divisible by batch size")
    segments = sequence.split(batch_size, dim=dim)
    batched = torch.cat(segments[:-1], dim=0)
    return batched
    

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

def compute_eos_loss(predictions, eos_targets, mask):
        """
        Computes the end-of-sequence (EOS) loss using negative log likelihood (binary cross entropy).

        Args:
            predictions (torch.Tensor): The predicted logits of shape (batch_size, sequence_length, num_classes).
            eos_targets (torch.Tensor): The target EOS indices of shape (batch_size, sequence_length).
            mask (torch.Tensor): A mask tensor of shape (batch_size, sequence_length) indicating valid positions.

        Returns:
            torch.Tensor: The computed EOS loss.
        """
        # Apply log softmax to predictions
        log_softmax = torch.log_softmax(predictions, dim=-1)
        # Gather the log probabilities corresponding to the targets
        log_probs = log_softmax.gather(dim=-1, index=eos_targets.unsqueeze(-1)).squeeze(-1)
        # Apply mask to log probabilities
        log_probs = log_probs * mask
        # Calculate the negative log likelihood
        nll_loss = -log_probs.sum() / mask.sum()
        return nll_loss
