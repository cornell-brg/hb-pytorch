"""
Unit Test for kernel_ceil
Written by Kofi Efah(kae87) 5/17/2020
"""
import torch

torch.manual_seed(42)

def _test_torch_ceil_check(x):
    h = x.hammerblade()
    ceil_x = x.ceil()
    ceil_h = h.ceil()
    assert ceil_h.device == torch.device("hammerblade")
    assert torch.allclose(ceil_h.cpu(), ceil_x)

def test_torch_ceil_1():
    x = torch.ones(10)
    _test_torch_ceil_check(x)

def test_torch_ceil_2():
    x = torch.randn(10, 10)
    _test_torch_ceil_check(x)

def test_torch_ceil_3():
    x = torch.zeros(10)
    _test_torch_ceil_check(x)
