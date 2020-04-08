"""
Unit tests for log_softmax operator
03/20/2020 Bandhav Veluri
"""

import torch
import torch.nn.functional as F
import random
import pytest
from hypothesis import given, settings
import hypothesis.strategies as st
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_log_softmax(x, dim):
    x_hb = x.hammerblade()

    y = F.log_softmax(x, dim)
    y_hb = F.log_softmax(x_hb, dim)

    assert torch.allclose(y, y_hb.cpu(), atol=1e-7)

def test_log_softmax_1():
    x = torch.rand(2, 3)
    dim = 1
    _test_log_softmax(x, dim)

def test_log_softmax_2():
    x = torch.rand(2, 3)
    dim = 0
    _test_log_softmax(x, dim)

def test_log_softmax_3():
    x = torch.rand(5)
    dim = 0
    _test_log_softmax(x, dim)

def test_log_softmax_4():
    x = torch.rand(1, 6)
    dim = 0
    _test_log_softmax(x, dim)

def test_log_softmax_5():
    x = torch.rand(1, 6)
    dim = 1
    _test_log_softmax(x, dim)

def test_log_softmax_6():
    x = torch.rand(2, 3, 3, 5)

    for dim in range(4):
        _test_log_softmax(x, dim)

@pytest.mark.skip(reason="known failure #61")
def test_log_softmax_large_1d():
    x = torch.tensor([88.72284])
    dim = 0
    _test_log_softmax(x, dim)

@pytest.mark.skip(reason="known failure #61")
@settings(deadline=None)
@given(tensor=hu.tensor())
def test_log_softmax_hypothesis(tensor):
    x = torch.tensor(tensor)

    for dim in range(x.dim()):
        _test_log_softmax(x, dim)
