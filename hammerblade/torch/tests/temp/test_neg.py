"""
Unit tests for torch.neg kernel
04/23/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import random
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_torch_neg(x):
    h = x.hammerblade()
    neg_x = -x
    neg_h = -h
    assert neg_h.device == torch.device("hammerblade")
    assert torch.equal(neg_h.cpu(), neg_x)

def test_torch_neg_1():
    x = torch.ones(10)
    _test_torch_neg(x)

def test_torch_neg_2():
    x = torch.randn(3, 4)
    _test_torch_neg(x)

@settings(deadline=None)
@given(tensor=hu.tensor())
def test_torch_neg_hypothesis(tensor):
    x = torch.tensor(tensor)
    _test_torch_neg(x)
