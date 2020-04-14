"""
Tests on torch.floor (floor kernel)
04/10/2020 Michelle Chao (mc2244@cornell.edu)
"""
import torch

torch.manual_seed(42)

def _test_torch_floor_check(x):
    h = x.hammerblade()
    floor_x = x.floor()
    floor_h = h.floor()
    assert floor_h.device == torch.device("hammerblade")
    assert torch.allclose(floor_h.cpu(), floor_x)

def test_torch_floor_1():
    x = torch.ones(10)
    _test_torch_floor_check(x)

def test_torch_floor_2():
    x = torch.randn(16, 16)
    _test_torch_floor_check(x)

def test_torch_floor_3():
    x = torch.randn(1)
    _test_torch_floor_check(x)

def test_torch_floor_4():
    x = torch.randn(10)
    _test_torch_floor_check(x)
    