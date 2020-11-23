"""
tests for hardtanh
Bandhav Veluri
"""

import torch

def _test_hardtanh_(x, min, max):
    h = x.hammerblade()
    assert h is not x
    out_cpu = torch.nn.functional.hardtanh(x, min, max)
    out = torch.nn.functional.hardtanh(h, min, max)
    assert out.is_hammerblade
    assert torch.allclose(out.cpu(), out_cpu)

def _test_hardtanh(x):
    _test_hardtanh_(x, 0, 0)
    _test_hardtanh_(x, -0.5, 0.5)
    _test_hardtanh_(x, -1, 1)

def test_hardtanh_1():
    a = torch.randn(10)
    _test_hardtanh(a)

def test_hardtanh_2():
    a = torch.rand(1, 128)
    _test_hardtanh(a)

def test_hardtanh_3():
    a = torch.rand(16, 32)
    _test_hardtanh(a)
