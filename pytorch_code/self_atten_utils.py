
# pytorch port of tf code
# pytorch default ordering of data is different
# we assume that we have B, d, H, W

import torch

def shape_list(x: torch.Tensor) -> list:
    """Return a list of dims"""
    return list(x.shape)


def split_heads_2d(inputs: torch.Tensor, Nh: int) -> torch.Tensor:
    """Split channels into multiple heads"""
    """Reorder to B, d, H, W of Pytorch"""

    B, d, H, W = shape_list(inputs)
    ret_shape = [B, H, W, Nh, d // Nh]
    split = torch.reshape(inputs, ret_shape)
    return split.permute(0, 3, 1, 2, 4)


def combine_heads_2d(inputs: torch.Tensor) -> torch.Tensor:
    """Combine heads (inverse of split heads 2d)."""
    """for now assume same format as tf"""
    transposed = inputs.permute(0,2,3,1,4)
    Nh, channels = shape_list(transposed)[-2:]
    ret_shape = shape_list(transposed)[:-2] + [Nh * channels]
    return torch.reshape(transposed, ret_shape) 

def rel_to_abs(x):
    """Converts tensor from relative to absolute indexing."""
    """why do we need this?"""
    
    B, Nh, L, _ = shape_list(x)
    col_pad = torch.zeros([B, Nh, L, 1])

    x = torch.cat((x, col_pad), dim=3)
    


