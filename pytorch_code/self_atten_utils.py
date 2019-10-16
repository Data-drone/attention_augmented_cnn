
# pytorch port of tf code

import torch

def shape_list(x: torch.Tensor):
    """Return a list of dims"""
    return list(x.shape)
    