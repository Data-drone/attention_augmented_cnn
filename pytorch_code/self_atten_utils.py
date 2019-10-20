
# pytorch port of tf code

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