"""
Unit tests for torch.dot kernel
03/06/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import pytest
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

def test_torch_dot_1():
    x = torch.ones(10)
    x_h = x.hammerblade()
    x = x.dot(x)
    x_h = x_h.dot(x_h)
    assert x_h.device == torch.device("hammerblade")
    assert torch.equal(x_h.cpu(), x)

def test_torch_dot_2():
    x = torch.randn(10)
    x_h = x.hammerblade()
    x = x.dot(x)
    x_h = x_h.dot(x_h)
    assert x_h.device == torch.device("hammerblade")
    assert torch.equal(x_h.cpu(), x)

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
    def dot(inputs):
        x1, x2 = inputs
        return x1.dot(x2)
    hu.assert_hb_checks(dot, inputs)
