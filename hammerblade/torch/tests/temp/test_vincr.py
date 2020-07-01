"""
Unit tests for torch.vincr [tutorial kernel]
06/11/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import random
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_torch_vincr(tensor):
    h = tensor.hammerblade()
    out = torch.vincr(h)
    assert out.device == torch.device("hammerblade")
    assert torch.allclose(tensor + 1, out.cpu())

def test_torch_vincr_1():
    t = torch.ones(10)
    _test_torch_vincr(t)

def test_torch_vincr_2():
    t = torch.randn(10)
    _test_torch_vincr(t)

def test_torch_vincr_3():
    t = torch.randn(2, 3)
    _test_torch_vincr(t)

@settings(deadline=None)
@given(tensor=hu.tensor())
def test_torch_vincr_hypothesis(tensor):
    t = torch.tensor(tensor)
    _test_torch_vincr(t)
