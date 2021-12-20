"""
Unit tests for torch.max
11/14/2021 Aditi Agarwal (aa2224@cornell.edu)
"""

import torch
import random
import pytest
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)


# ------------------------------------------------------------------------
# test of getting max_value of tensor t along dimension dim
# ------------------------------------------------------------------------

def _test_torch_max_val(t, dim, keepdim=False):
    h = t.hammerblade()
    out = torch.max(t,dim, keepdim=keepdim)
    print("CPU output:")
    print(out)
    print("\n")
    out_h = torch.max(h,dim, keepdim=keepdim)
    print("hammerblade output:")
    print(out_h.cpu())
    print("\n")
    assert out_h.device == torch.device("hammerblade")
    assert torch.allclose(out_h.cpu(), out)   


def test_max_val_1():
    t = torch.ones(10)
    _test_torch_max_val(t,0)

def test_max_val_rand0():
    t = torch.randn(4)
    _test_torch_max_val(t,0)

def test_max_val_rand2x():
    t = torch.randn(4,5)
    _test_torch_max_val(t,0, keepdim=True)

def test_max_val_rand2x2():
    t = torch.randn(10,20)
    _test_torch_max_val(t,0)

def test_max_val_rand2y():
    t = torch.randn(10,20)
    _test_torch_max_val(t,1, keepdim=True)

def test_max_val_rand2y2():
    t = torch.randn(4,5)
    _test_torch_max_val(t,1)

def test_max_val_rand3x():
    t = torch.randn(4,5, 6)
    _test_torch_max_val(t,0)

def test_max_val_rand3x2():
    t = torch.randn(4,5, 6)
    _test_torch_max_val(t,0, keepdim=True)

def test_max_val_rand3y():
    t = torch.randn(4,5, 6)
    _test_torch_max_val(t,1)

def test_max_val_rand3y2():
    t = torch.randn(4,5, 6)
    _test_torch_max_val(t,1, keepdim=True)

def test_max_val_rand3z():
    t = torch.randn(4,5, 6)
    _test_torch_max_val(t,0, keepdim=True)
