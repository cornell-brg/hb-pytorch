"""
Tests on torch.nn.relu (threshold kernel)
01/23/2020 Lin Cheng (zz546@cornell.edu)
"""
import torch
import torch.nn as nn
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

def test_torch_nn_sign_1():
    x = torch.ones(10)
    x_h = x.hammerblade()
    x_sign = torch.sign(x)
    x_h_sign = torch.sign(x_h)
    assert x_h_sign.device == torch.device("hammerblade")
    assert torch.allclose(x_h_sign.cpu(), x_sign)

def test_torch_nn_sign_2():
    x = torch.randn(10)
    x_h = x.hammerblade()
    x_sign = torch.sign(x)
    x_h_sign = torch.sign(x_h)
    assert x_h_sign.device == torch.device("hammerblade")
    assert torch.allclose(x_h_sign.cpu(), x_sign)

def test_torch_nn_sign_3():
    x = torch.randn(10)
    x_h = x.hammerblade()
    x.sign_()
    x_h.sign_()
    assert x_h.device == torch.device("hammerblade")
    assert torch.allclose(x_h.cpu(), x)

def test_torch_nn_sigmoid_3():
    x = torch.randn(3, 4)
    x_h = x.hammerblade()
    x.sign_()
    x_h.sign_()
    assert x_h.device == torch.device("hammerblade")
    assert torch.allclose(x_h.cpu(), x)

def _test_torch_sigmoid_check(tensor_self):
    tensor_self_hb = torch.tensor(tensor_self).hammerblade()
    result_hb = torch.sign(tensor_self_hb)
    assert result_hb.device == torch.device("hammerblade")
    assert torch.allclose(result_hb.cpu(), torch.sign(torch.tensor(tensor_self)))

@settings(deadline=None)
@given(tensor=hu.tensor())
def test_elementwise_torch_sigmoid_hypothesis(tensor):
    _test_torch_sigmoid_check(tensor)
