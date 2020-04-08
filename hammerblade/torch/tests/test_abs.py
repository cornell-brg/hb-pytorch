"""
Unit tests for torch.abs kernel
03/06/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import random
from hypothesis import given, settings
import hypothesis.strategies as st
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_torch_asb(x):
    h = x.hammerblade()
    abs_x = x.abs()
    abs_h = h.abs()
    assert abs_h.device == torch.device("hammerblade")
    assert torch.equal(abs_h.cpu(), abs_x)

def test_torch_abs_1():
    x = torch.ones(10)
    _test_torch_asb(x)

def test_torch_abs_2():
    x = torch.randn(3, 4)
    _test_torch_asb(x)

@settings(deadline=None)
@given(tensor=hu.tensor())
def test_torch_abs_hypothesis(tensor):
    x = torch.tensor(tensor)
    _test_torch_asb(x)
