"""
Tests on LLCopy
05/01/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import random

torch.manual_seed(42)
random.seed(42)

def _compare(x, h):
    assert x.numel() == h.numel()
    assert x.element_size() == h.element_size()
    assert x.dtype == h.dtype
    assert x.stride() == h.stride()
    assert x.storage_offset() == h.storage_offset()
    assert x.size() == h.size()
    assert x.is_contiguous() == h.is_contiguous()
    assert h.device == torch.device("hammerblade")
    assert torch.equal(h.cpu(), x)

def test_llcopy_1():
    x = torch.ones(10)
    h = x.hammerblade_ll()
    _compare(x, h)

def test_llcopy_2():
    x = torch.ones(10)[2]
    h = x.hammerblade_ll()
    _compare(x, h)

def test_llcopy_3():
    x = torch.ones(3, 4).t()
    h = x.hammerblade_ll()
    _compare(x, h)

def test_llcopy_4():
    x = torch.randn(3, 4).t()
    h = x.hammerblade_ll()
    _compare(x, h)

def test_llcopy_5():
    x = torch.randn(3, 4).t()[2]
    h = x.hammerblade_ll()
    _compare(x, h)

def test_normal_copy():
    x = torch.randn(3, 4).t()[2]
    h = x.hammerblade()
    assert not x.is_contiguous()
    assert h.is_contiguous()
