"""
Hypothesis property-based random testing
helper functions adapted from caffe2/python/hypothesis_test_util.py
"""

import torch
import copy
import hypothesis
import hypothesis.extra.numpy
import hypothesis.strategies as st
import numpy as np


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
        elems = st.floats(min_value=-1.0, max_value=1.0, width=32)
    # elif dtype is np.int32:
    #     elems = st.integers(min_value=0, max_value=2 ** 31 - 1)
    # elif dtype is np.bool:
    #     elems = st.booleans()
    else:
        raise ValueError("Unexpected dtype without elements provided")
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
    assert len(output1) == len(output2)
    for i in range(len(output1)):
        o1 = move_to_cpu(output1[i])
        o2 = move_to_cpu(output2[i])
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
                lambda dims: arrays(dims, dtype, elements, filter_=(lambda x: x!=0)))
        else:
            return dims_.flatmap(
                lambda dims: arrays(dims, dtype, elements))


    @staticmethod
    def tensor1d(min_len=1, max_len=64, dtype=np.float32, elements=None):
        return tensor(1, 1, dtype, elements, min_value=min_len, max_value=max_len)


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
                lambda dims: st.lists(
                  arrays(dims, dtype, elements, filter_=(lambda x: x!=0)),
                    min_size=n,
                    max_size=n))
        else:
            return dims_.flatmap(
                lambda dims: st.lists(
                    arrays(dims, dtype, elements),
                    min_size=n,
                    max_size=n))


    @staticmethod
    def tensors1d(n, min_len=1, max_len=64, dtype=np.float32, elements=None):
        return tensors(
            n, 1, 1, dtype, elements, min_value=min_len, max_value=max_len
        )


    # ===========================
    # Result comparison
    # ===========================

    # run the same op on both cpu and HB
    @staticmethod
    def assert_hb_checks(op, inputs):
        inputs_h = []
        for input in inputs:
            inputs_h.append(torch.tensor(input).hammerblade())
        inputs_c = []
        for input in inputs:
            inputs_c.append(torch.tensor(input))
        outputs_cpu = op(inputs_c)
        outputs_hb = op(inputs_h)
        for output in outputs_hb:
            assert output.device == torch.device("hammerblade")
        assert_equal(outputs_cpu, outputs_hb)
