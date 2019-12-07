
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
    flat_x = torch.reshape(x, [B, Nh, L * 2 * L])
    flat_pad = torch.zeros([B, Nh, L-1])
    flat_x_pad = torch.concat([flat_x, flat_pad], axis = 2)

    final_x = torch.reshape(flat_x_pad, [B, Nh, L+1, 2*L-1])
    final_x = final_x[:, :, :L, L-1:]
    return final_x


def relative_logits_1d(q, rel_k, H, W, Nh, transpose_mask):
    """Compute relative logits along one dimension."""
    """Need to document inputs to make sure we test right"""

    rel_logits = torch.einsum('bhxyd,md->bhxym', q, rel_k)
    # Collapse height and heads
    rel_logits = torch.reshape(
        rel_logits, [-1, Nh * H, W, 2 * W-1]
    )
    rel_logits = rel_to_abs(rel_logits)
    # Shape it and tile height times
    rel_logits = torch.reshape(rel_logits, [-1, Nh, H, W, W])
    rel_logits = torch.expand_dims(rel_logits, axis=3)
    rel_logits = torch.tile(rel_logits, [1, 1, 1, H, 1, 1])
    # Reshape for adding to the logits.
    rel_logits = torch.transpose(rel_logits, transpose_mask)
    rel_logits = torch.reshape(rel_logits, [-1, Nh, H*W, H*W])
    return rel_logits