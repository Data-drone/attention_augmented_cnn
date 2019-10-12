
# direct copy of the code from:
# https://arxiv.org/pdf/1904.09925.pdf
# page 12
# conv layer

import tensorflow as tf
from .self_atten_utils_tf import relative_logits_1d, shape_list, \
                                split_heads_2d, combine_heads_2d

def relative_logits(q, H, W, Nh, dkh):
    """Compute relative logits."""
    # Relative logits in width dimension first.
    rel_embeddings_w = tf.get_variable(
        'r_width', shape=(2*W - 1, dkh),
        initializer = tf.random_normal_initializer(dkh**-0.5)
    )

    # [B, Nh, HW, HW]
    rel_logits_w = relative_logits_1d(
        q, rel_embeddings_w, H, W, Nh, [0, 1, 2, 4, 3, 5]
    )

    # Relative logits in height dimension next.
    # For ease, we 1) transpose height and width,
    # 2) repeat the above steps and
    # 3) transpose to eventually put the logits
    # in their right positions.

    rel_embeddings_h = tf.get_variable(
        'r_height', shape=(2 * H - 1, dkh),
        initializer = tf.random_normal_initializer(dkh**-0.5)
    )
    # [B, Nh, HW, HW]
    rel_logits_h = relative_logits_1d(
        tf.transpose(q, [0, 1, 3, 2, 4]),
        rel_embeddings_h, W, H, Nh, [0, 1, 4, 2, 5, 3]
    )

    return rel_logits_h, rel_logits_w


def self_attention_2d(inputs, dk, dv, Nh, relative=True):
    """2d relative self-attention."""
    _, H, W, _ = shape_list(inputs)
    dkh = dk // Nh
    dvh = dv // Nh
    flatten_hw = lambda x, d: tf.reshape(x, [-1, Nh, H*W, d])

    # Compute q, k, v
    kqv = tf.layers.conv2d(inputs, 2 * dk + dv, 1)
    k, q, v = tf.split(kqv, [dk, dk, dv], axis=3)
    q *= dkh ** -0.5 # scaled dot-product

    # After splitting, shape = [B, Nh, H, W, dkh or dvh]
    q = split_heads_2d(q, Nh)
    k = split_heads_2d(k, Nh)
    v = split_heads_2d(v, Nh)

    # [B, Nh, HW, HW]
    logits = tf.matmul(flatten_hw(q, dkh), flatten_hw(k, dkh),
        transpose_b=True)

    if relative:
        rel_logits_h, rel_logits_w = relative_logits(q, H, W, Nh,
                dkh)
        logits += rel_logits_h
        logits += rel_logits_w

    weights = tf.nn.softmax(logits)
    attn_out = tf.matmul(weights, flatten_hw(v, dvh))
    attn_out = tf.reshape(attn_out, [-1, Nh, H, W, dvh])
    attn_out = combine_heads_2d(attn_out)
    # Project heads
    attn_out = tf.layers.conv2d(attn_out, dv, 1)
    return attn_out

def augmented_conv2d(X, Fout, k, dk, dv, Nh, relative):
    conv_out = tf.layers.conv2d(inputs=X, filters=Fout - dv,
        kernel_size=k, padding='same')
    attn_out = self.attention_2d(X, dk, dv, Nh, relative=relative)
    return tf.concat([conv_out, attn_out], axis=3)

