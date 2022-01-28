"""
BRG tests on PyTorch => tests of real offloading kernels
May 10, 2021
Zhongyuan Zhao
"""

import torch
import random
from math import isnan, isinf
from hypothesis import assume, given, settings
import hypothesis.strategies as st
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

# test of min two tensors

def _test_min(x1, x2):
    h1 = x1.hammerblade()
    h2 = x2.hammerblade()
    assert h1 is not x1
    assert h2 is not x2
    y_c = torch.min(x1, x2)
    y_h = torch.min(h1, h2)
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_c, y_h.cpu())

def test_min_1():
    x1 = torch.zeros(1, 10)
    x2 = torch.ones(1, 10)
    _test_min(x1, x2)

def test_min_2():
    x1 = torch.rand(1, 128)
    x2 = torch.rand(1, 128)
    _test_min(x1, x2)

def test_min_3():
    x1 = torch.rand(16, 32)
    x2 = torch.rand(16, 32)
    _test_min(x1, x2)

def test_min_4():
    x1 = torch.rand(16, 32)
    x2 = torch.rand(32)
    _test_min(x1, x2)

def test_min_5():
    x1 = torch.rand(32)
    x2 = torch.rand(16, 32)
    _test_min(x1, x2)

