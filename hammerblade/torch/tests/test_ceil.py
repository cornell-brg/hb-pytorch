"""
Unit Test for kernel_ceil
Written by Kofi Efah(kae87) 4/13/2020
"""
import torch
import random

torch.manual_seed(42)
random.seed(42)

def _ceil_test_helper(x):
    h = x.hammerblade()
    ceil_x = x.ceil()
    ceil_h = h.ceil()
    assert ceil_h.device == torch.device("hammerblade")
    assert torch.equal(ceil_h.cpu(),ceil_x)

def test_torch_ceil_1():
    x = torch.ones(100)
    _ceil_test_helper(x)

def test_torch_ceil_2():
    x = torch.randn(2, 9)
    _ceil_test_helper(x)

def test_torch_ceil_3():
    x = torch.zeros(100)
    _ceil_test_helper(x)


