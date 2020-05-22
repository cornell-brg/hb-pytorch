"""
Unit tests for torch.fill_ kernel
03/06/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import random
import hypothesis.strategies as st
from math import isnan, isinf
from hypothesis import assume, given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_torch_fill(x, s):
    h = x.hammerblade()
    x.fill_(s)
    h.fill_(s)
    assert h.device == torch.device("hammerblade")
    assert torch.equal(h.cpu(), x)

def test_torch_fill_1():
    x = torch.empty(1, 10)
    _test_torch_fill(x, 42)

def test_torch_fill_2():
    x = torch.empty(5, 6)
    _test_torch_fill(x, 42)

@settings(deadline=None)
@given(tensor=hu.tensor(), scalar=st.floats(width=32))
def test_torch_fill_hypothesis(tensor, scalar):
    assume(scalar != 0)
    assume(not isnan(scalar))
    assume(not isinf(scalar))
    x = torch.tensor(tensor)
    _test_torch_fill(x, scalar)
