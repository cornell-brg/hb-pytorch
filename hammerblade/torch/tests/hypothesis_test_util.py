"""
Hypothesis property-based random testing
helper functions adapted from caffe2/python/hypothesis_test_util.py
"""

import torch
import hypothesis
import hypothesis.extra.numpy
import hypothesis.strategies as st
import numpy as np
import os
import random

def is_travis():
    return 'TRAVIS' in os.environ


# ===========================
# Tensor construction helpers
# ===========================

def dims(min_value=1, max_value=5):
    return st.integers(min_value=min_value, max_value=max_value)

def elements_of_type(dtype=np.float32, filter_=None):
    elems = None
    if dtype is np.float32:
        if is_travis():
            elems = st.floats(min_value=-2.0, max_value=2.0, width=32)
        else:
            elems = st.floats(min_value=-128.0, max_value=128.0, width=32)
    # elif dtype is np.int32:
    #     elems = st.integers(min_value=0, max_value=2 ** 31 - 1)
    # elif dtype is np.bool:
    #     elems = st.booleans()
    else:
        raise ValueError("Unexpected dtype without elements provided")
    elems = elems.filter(lambda x: x > 0.001 or x < -0.001)
    return elems if filter_ is None else elems.filter(filter_)


def arrays(dims, dtype=np.float32, elements=None, filter_=None):
    if elements is None:
        elements = elements_of_type(dtype, filter_)
    return hypothesis.extra.numpy.arrays(
        dtype,
        dims,
        elements=elements,
    )


def move_to_cpu(tensor):
    if tensor.is_hammerblade:
        return tensor.cpu()
    else:
        return tensor


def assert_equal(output1, output2):
    o1 = move_to_cpu(output1)
    o2 = move_to_cpu(output2)
    assert torch.allclose(o1, o2)


class HypothesisUtil():

    @staticmethod
    def tensor(min_dim=1,
               max_dim=4,
               dtype=np.float32,
               elements=None,
               nonzero=False,
               **kwargs):
        dims_ = st.lists(dims(**kwargs), min_size=min_dim, max_size=max_dim)
        if nonzero:
            return dims_.flatmap(
                lambda dims: arrays(dims, dtype, elements, filter_=(lambda x: x != 0)))
        else:
            return dims_.flatmap(
                lambda dims: arrays(dims, dtype, elements))


    @staticmethod
    def tensor1d(min_len=1, max_len=64, dtype=np.float32, elements=None, nonzero=False):
        return HypothesisUtil.tensor(1, 1, dtype, elements, nonzero, min_value=min_len, max_value=max_len)


    @staticmethod
    def tensors(n,
                min_dim=1,
                max_dim=4,
                dtype=np.float32,
                elements=None,
                nonzero=False,
                **kwargs):
        dims_ = st.lists(dims(**kwargs), min_size=min_dim, max_size=max_dim)
        if nonzero:
            return dims_.flatmap(
                lambda dims: st.lists(arrays(dims, dtype, elements, filter_=(lambda x: x != 0)), min_size=n, max_size=n))
        else:
            return dims_.flatmap(
                lambda dims: st.lists(arrays(dims, dtype, elements), min_size=n, max_size=n))

    @staticmethod
    @st.composite
    def tensors2dsquare(draw, min_shape=1, max_shape=10, dtype=np.float32):
        # create 2d square matrices
        s = st.integers(min_shape, max_shape)
        a = st.shared(s, key="hi")
        b = st.shared(s, key="hi")
        shapes = st.tuples(a,b)
        return draw(arrays(shapes, dtype=dtype))

    @staticmethod
    def tensors1d(n, min_len=1, max_len=64, dtype=np.float32, elements=None, nonzero=False):
        return HypothesisUtil.tensors(
            n, 1, 1, dtype, elements, nonzero, min_value=min_len, max_value=max_len
        )
