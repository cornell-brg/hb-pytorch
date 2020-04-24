"""
Tests for exp
04/13/2020 Yuyi He (yh383@cornell.edu)
"""
import torch

torch.manual_seed(42)

def _test_torch_exp(x):
    h = x.hammerblade()
    exp_x = x.exp()
    exp_h = h.exp()
    assert exp_h.device == torch.device("hammerblade")
    assert torch.allclose(exp_h.cpu(), exp_x)

def _test_torch_exp_1():
    x = torch.randn(3, 5)
    _test_torch_exp(x)

def _test_torch_exp_2():
    x = torch.ones(10)
    _test_torch_exp(x)

def _test_torch_exp_3():
    x = torch.zeros(5)
    _test_torch_exp(x)
