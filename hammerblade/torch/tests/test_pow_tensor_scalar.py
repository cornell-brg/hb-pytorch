"""
Tests on pow_tensor_scalar (tensor base, scalar exp)
08/17/2021 Niklas Schmelzle (jms854@cornell.edu)
"""
import torch
import torch.nn as nn
from random import randrange
import numpy as np

def _test_pow_tensor_scalar(inp_list, exp_scalar):
    inp = torch.tensor(inp_list)
    inph = inp.hammerblade()

    out = inp ** exp_scalar
    outh = inph ** exp_scalar
    print(out)
    print(outh)

    assert outh.device == torch.device("hammerblade")
    # NaN results when base negative and exp < 1
    assert torch.allclose(out, outh.cpu(), equal_nan=True)


def test_scalar_scalar():
    inp_scalar = np.random.rand(1).tolist()
    for exp in [0.5, 2.0, 3.0, -0.5, -1.0, -2.0, randrange(42)]:
        _test_pow_tensor_scalar([42.42], exp)
        _test_pow_tensor_scalar(inp_scalar, exp)

def test_tensor_scalar():
    list_size = randrange(1, 1000)
    inp_list = np.random.rand(list_size).tolist()
    for exp in [0.5, 2.0, 3.0, -0.5, -1.0, -2.0, randrange(42)]:
        _test_pow_tensor_scalar([-3.0, -0.5, 0.0, 0.3, 1.7, 17.0, 42.42, 128.0], exp)
        _test_pow_tensor_scalar(inp_list, exp)
