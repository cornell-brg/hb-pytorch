"""
BRG tests on PyTorch => tests of real offloading kernels
Feb 09, 2020
Lin Cheng
"""

import torch
import random
from math import isnan, isinf
from hypothesis import assume, given, settings
import hypothesis.strategies as st
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

# test of max two tensors

def _test_max(x1, x2):
    h1 = x1.hammerblade()
    h2 = x2.hammerblade()
    assert h1 is not x1
    assert h2 is not x2
    y_c = torch.max(x1, x2)
    y_h = torch.max(h1, h2)
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_c, y_h.cpu())

def test_max_1():
    x1 = torch.zeros(1, 10)
    x2 = torch.ones(1, 10)
    _test_max(x1, x2)

def test_max_2():
    x1 = torch.rand(1, 128)
    x2 = torch.rand(1, 128)
    _test_max(x1, x2)

def test_max_3():
    x1 = torch.rand(16, 32)
    x2 = torch.rand(16, 32)
    _test_max(x1, x2)

