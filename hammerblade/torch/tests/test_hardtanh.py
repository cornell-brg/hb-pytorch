"""
tests for hardtanh
Bandhav Veluri
"""
import torch

def _test_hardtanh_(x, min, max, inplace=False):
    h = x.hammerblade()
    out_cpu = torch.nn.functional.hardtanh(x, min,
                                           max, inplace)
    out = torch.nn.functional.hardtanh(h, min,
                                       max, inplace)
    assert torch.allclose(out.cpu(), out_cpu)
    assert torch.allclose(h.cpu(), x)

def _test_hardtanh(x):
    for inplace in [True, False]:
        _test_hardtanh_(x, 0, 0, inplace)
        _test_hardtanh_(x, -0.5, 0.5, inplace)
        _test_hardtanh_(x, -1, 1, inplace)

def test_hardtanh_1():
    a = torch.randn(10)
    _test_hardtanh(a)

def test_hardtanh_2():
    a = torch.rand(1, 128)
    _test_hardtanh(a)

def test_hardtanh_3():
    a = torch.rand(16, 32)
    _test_hardtanh(a)
