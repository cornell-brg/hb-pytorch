"""
Tests on select
04/22/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import random

torch.manual_seed(42)
random.seed(42)

def _test_select(x, index):
    h = x.hammerblade()
    assert h[index].device == torch.device("hammerblade")
    assert torch.equal(x[index], h[index].cpu())

def _test_select2d(x, i, j):
    h = x.hammerblade()
    assert h[i, j].device == torch.device("hammerblade")
    assert torch.equal(x[i, j], h[i, j].cpu())

def test_select_1():
    x = torch.ones(10)
    _test_select(x, 0)
    _test_select(x, 5)
    _test_select(x, 9)

def test_select_2():
    x = torch.randn(10)
    for i in range(10):
        _test_select(x, i)

def test_select_3():
    x = torch.randn(3, 5)
    for i in range(3):
        _test_select(x, i)

def test_select_4():
    x = torch.randn(3, 5)
    for i in range(3):
        _test_select(x, i)
        for j in range(5):
            _test_select2d(x, i, j)
