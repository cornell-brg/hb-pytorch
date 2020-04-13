"""
Unit tests for torch.dot kernel
03/06/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import pytest
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)

def _test_torch_dot(x1, x2):
    h1 = x1.hammerblade()
    h2 = x2.hammerblade()
    y_c = x1.dot(x2)
    y_h = h1.dot(h2)
    assert y_h.device == torch.device("hammerblade")
    assert torch.equal(y_h.cpu(), y_c)

def test_torch_dot_1():
    x = torch.ones(10)
    _test_torch_dot(x, x)

def test_torch_dot_2():
    x1 = torch.randn(10)
    x2 = torch.randn(10)
    _test_torch_dot(x1, x2)

def test_torch_dot_non_contiguous():
    x = torch.tensor([1.])
    x_h = x.hammerblade()
    x = x.expand(10)
    x_h = x_h.expand(10)
    x = x.dot(x)
    x_h = x_h.dot(x_h)
    assert x_h.device == torch.device("hammerblade")
    assert torch.equal(x_h.cpu(), x)

@pytest.mark.xfail
def test_torch_dot_different_device_F():
    x = torch.ones(10)
    x_h = x.hammerblade()
    x_h = x_h.dot(x)

@pytest.mark.xfail
def test_torch_dot_mismatching_shape_F():
    x = torch.ones(10).hammerblade()
    y = torch.ones(5).hammerblade()
    torch.dot(x, y)

@settings(deadline=None)
@given(inputs=hu.tensors1d(n=2))
def test_torch_dot_hypothesis(inputs):
    x1 = torch.tensor(inputs[0])
    x2 = torch.tensor(inputs[1])
    _test_torch_dot(x1, x2)
