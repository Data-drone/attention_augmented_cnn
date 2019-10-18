# utils to test torch vs tf

import unittest
import torch
import tensorflow as tf

from pytorch_code.self_atten_utils import shape_list as py_shape_list
from tf_code.self_atten_utils_tf import shape_list as tf_shape_list

def entry_shape_list(test_shape):
    
    

    tf_tensor = tf.random.normal(
        shape = test_shape
    )

    torch_tensor = torch.rand(test_shape)

    assert py_shape_list(torch_tensor) == tf_shape_list(tf_tensor)


def test_shape_list():

    entry_shape_list([10, 60, 60, 3])
    entry_shape_list([60, 60, 3])
