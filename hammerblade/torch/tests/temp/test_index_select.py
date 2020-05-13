"""
Tests on torch.index_select
04/22/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import random

torch.manual_seed(42)
random.seed(42)

def _test_torch_index_select(x, index, dim):
    h = x.hammerblade()
    out = torch.index_select(x, dim, index)
    out_h = torch.index_select(h, dim, index.hammerblade())
    assert out_h.device == torch.device("hammerblade")
    assert torch.equal(out, out_h.cpu())

def test_torch_index_select_1():
    x = torch.ones(3, 4)
    index = torch.tensor([0, 2])
    _test_torch_index_select(x, index, 0)
    _test_torch_index_select(x, index, 1)

def test_torch_index_select_2():
    x = torch.randn(3, 4)
    index = torch.tensor([0, 2])
    _test_torch_index_select(x, index, 0)
    _test_torch_index_select(x, index, 1)

def test_torch_index_select_3():
    x = torch.randn(3, 4, 5)
    index = torch.tensor([0, 2])
    _test_torch_index_select(x, index, 0)
    _test_torch_index_select(x, index, 1)
    _test_torch_index_select(x, index, 2)
