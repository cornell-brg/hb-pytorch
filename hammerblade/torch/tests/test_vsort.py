"""
[TUTORIAL]
Unit tests for torch.vsort kernel
06/12/2020 Krithik Ranjan (kr397@cornell.edu)
"""

import torch
import random
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_torch_vsort(t1):
    h1 = t1.hammerblade()
    out, indices = torch.sort(t1)
    out_h = torch.vsort(h1)
    assert out_h.device == torch.device('hammerblade')
    assert torch.allclose(out_h.cpu(), out)

def test_torch_vsort_1():
    # Trivial test for a vector with all ones
    t1 = torch.ones(10)
    _test_torch_vsort(t1)

def test_torch_vsort_2():
    # Fixed length, random elements
    t1 = torch.randn(15)
    _test_torch_vsort(t1)

def test_torch_vsort_3():
    # Fixed length exponent of 2, random elements
    t1 = torch.randn(16)
    _test_torch_vsort(t1)

def test_torch_vsort_4():
    # Random length, random elements
    size = random.randint(10, 100)
    t1 = torch.randn(size)
    _test_torch_vsort(t1)

@settings(deadline=None)
@given(tensor=hu.tensor1d(nonzero=True, min_len=5))
def test_torch_vsort_hypothesis(tensor):
    t1 = torch.tensor(tensor)
    _test_torch_vsort(t1)
