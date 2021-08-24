"""
tests of add and sub kernel with float, int, and long long elements.
Authors : Lin Cheng, Janice Wei
Date    : 02/09/2020, 07/13/2020
"""

import torch
import random
from math import isnan, isinf
from hypothesis import assume, given, settings
import hypothesis.strategies as st
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

# ------------------------------------------------------------------------
# test of adding two tensors x1 and x2
# ------------------------------------------------------------------------

def _test_add(x1, x2):
    h1 = x1.hammerblade()
    h2 = x2.hammerblade()
    assert h1 is not x1
    assert h2 is not x2
    y_c = x1 + x2
    y_h = h1 + h2
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_c, y_h.cpu())
    # inplace
#    x1.add_(x2)
#    h1.add_(h2)
#    assert h1.device == torch.device("hammerblade")
#    assert torch.allclose(x1, h1.cpu())

# ------------------------------------------------------------------------
# tests of add kernel with float elements
# ------------------------------------------------------------------------

#def test_add_1():
#    x = torch.rand(64)
#    _test_add(x, x)

def test_add_2():
    x1 = torch.rand(256, 32)
    x2 = torch.rand(256, 32)
    _test_add(x1, x2)

#def test_add_3():
#    x1 = torch.rand(32, 2, 32)
#    x2 = torch.rand(32, 2, 32)
#    _test_add(x1, x2)

#def test_relu_1():
#    x = torch.rand(64)
#    x_h = x.hammerblade()
#    relu = torch.nn.ReLU()
#    x_relu = relu(x)
#    x_h_relu = relu(x_h)
#    assert x_h_relu.device == torch.device("hammerblade")
#    assert torch.equal(x_h_relu.cpu(), x_relu)

#def test_relu_2():
#    x = torch.rand(32, 32)
#    x_h = x.hammerblade()
#    relu = torch.nn.ReLU()
#    x_relu = relu(x)
#    x_h_relu = relu(x_h)
#    assert x_h_relu.device == torch.device("hammerblade")
#    assert torch.equal(x_h_relu.cpu(), x_relu)

#def test_relu_3():
#    x = torch.rand(32, 12, 64)
#    x_h = x.hammerblade()
#    relu = torch.nn.ReLU()
#    x_relu = relu(x)
#    x_h_relu = relu(x_h)
#    assert x_h_relu.device == torch.device("hammerblade")
#    assert torch.equal(x_h_relu.cpu(), x_relu)
