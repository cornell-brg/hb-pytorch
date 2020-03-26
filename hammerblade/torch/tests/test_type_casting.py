"""
Tests on torch.to (copy_hb_to_hb kernel)
03/25/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import pytest

def test_torch_to_long_to_int_1():
    x = torch.ones(10, dtype=torch.long)
    assert x.type() == 'torch.LongTensor'
    h = x.hammerblade()
    int_x = x.to(torch.int)
    int_h = h.to(torch.int)
    assert int_h.type() == 'torch.hammerblade.IntTensor'
    assert int_h.device == torch.device("hammerblade")
    assert torch.equal(int_x, int_h.cpu())

def test_torch_to_long_to_int_2():
    x = torch.LongTensor(10).random_(0, 10)
    assert x.type() == 'torch.LongTensor'
    h = x.hammerblade()
    int_x = x.to(torch.int)
    int_h = h.to(torch.int)
    assert int_h.type() == 'torch.hammerblade.IntTensor'
    assert int_h.device == torch.device("hammerblade")
    assert torch.equal(int_x, int_h.cpu())

def test_torch_to_long_to_int_3():
    x = torch.LongTensor(64).random_(-2147483648, 2147483647)
    assert x.type() == 'torch.LongTensor'
    h = x.hammerblade()
    int_x = x.to(torch.int)
    int_h = h.to(torch.int)
    assert int_h.type() == 'torch.hammerblade.IntTensor'
    assert int_h.device == torch.device("hammerblade")
    assert torch.equal(int_x, int_h.cpu())

@pytest.mark.skip(reason="known bug")
def test_torch_to_int_to_long_1():
    x = torch.ones(10, dtype=torch.int)
    assert x.type() == 'torch.IntTensor'
    h = x.hammerblade()
    long_x = x.to(torch.long)
    long_h = h.to(torch.long)
    assert long_h.type() == 'torch.hammerblade.LongTensor'
    assert long_h.device == torch.device("hammerblade")
    assert torch.equal(long_x, long_h.cpu())
