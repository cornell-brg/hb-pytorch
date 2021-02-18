"""
Tests on torch.remainder (threshold kernel)
30/01/2021 Zhongyuan Zhao (zz546@cornell.edu)
"""
import torch
import torch.nn as nn
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

def test_torch_remainder_1():
    x = torch.IntTensor(10).random_()
    hb_x = x.hammerblade()
    y = torch.ones(1).int()
    hb_y = y.hammerblade()
    r = x % y
    hb_r = hb_x % hb_y
    assert hb_r.device == torch.device("hammerblade")
    assert torch.allclose(hb_r.cpu(), r)

def test_torch_remainder_2():
    x = torch.IntTensor(10).random_()
    hb_x = x.hammerblade()
    y = torch.ones(10).int() + 2
    hb_y = y.hammerblade()
    
    r = x % y
    hb_r = hb_x % hb_y
    assert hb_r.device == torch.device("hammerblade")
    assert torch.allclose(hb_r.cpu(), r)

def test_torch_nn_sigmoid_3():
    x = torch.IntTensor(3, 4).random_()
    hb_x = x.hammerblade()
    y = torch.ones(4).int() + 2
    hb_y = y.hammerblade()
    r = x % y
    hb_r = hb_x % hb_y
    assert hb_r.device == torch.device("hammerblade")
    assert torch.allclose(hb_r.cpu(), r)
