"""
Tests on Ceil kernel
05/19/2020 Kofi Efah (kae87)
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
    x = torch.randn(1)
    _test_torch_ceil_check(x)
