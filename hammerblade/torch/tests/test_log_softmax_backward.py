"""
Tests on torch.nn.LogSoftMax backward
03/26/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import torch.nn.functional as F
import random
import hbutils
import pytest
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_log_softmax_back(x, dim):
    x = x.clone().detach().requires_grad_(True)
    x_hb = hbutils.init_hb_tensor(x)
    assert x is not x_hb

    y = F.log_softmax(x, dim)
    y_hb = F.log_softmax(x_hb, dim)

    assert y_hb.device == torch.device("hammerblade")
    assert torch.allclose(y, y_hb.cpu(), atol=1e-7)

    y.backward(y.clone().detach() * -1.0)
    y_hb.backward(y_hb.clone().detach() * -1.0)

    assert x.grad is not None
    assert x_hb.grad is not None
    assert torch.allclose(x.grad, x_hb.grad.cpu(), atol=1e-6)

def test_log_softmax_back_1():
    x = torch.randn(2, 3)
    dim = 1
    _test_log_softmax_back(x, dim)

def test_log_softmax_back_2():
    x = torch.randn(2, 3)
    dim = 0
    _test_log_softmax_back(x, dim)

def test_log_softmax_back_3():
    x = torch.randn(5)
    dim = 0
    _test_log_softmax_back(x, dim)

def test_log_softmax_back_4():
    x = torch.randn(1, 6)
    dim = 1
    _test_log_softmax_back(x, dim)

def test_log_softmax_back_5():
    x = torch.randn(1, 6)
    dim = 0
    _test_log_softmax_back(x, dim)

def test_log_softmax_back_6():
    x = torch.randn(2, 3, 3, 5)

    for dim in range(4):
        _test_log_softmax_back(x, dim)

@pytest.mark.skip(reason="known failure #61")
@settings(deadline=None)
@given(tensor=hu.tensor())
def test_log_softmax_back_hypothesis(tensor):
    x = torch.tensor(tensor)

    for dim in range(x.dim()):
        _test_log_softmax_back(x, dim)
