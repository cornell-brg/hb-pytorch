"""
Unit tests for torch.fill_ kernel
03/06/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import hypothesis.strategies as st
from math import isnan, isinf
from hypothesis import assume, given, settings
from .hypothesis_test_util import HypothesisUtil as hu

def test_torch_fill_1():
    x = torch.empty(1, 10)
    x_h = x.hammerblade()
    x.fill_(42)
    x_h.fill_(42)
    assert x_h.device == torch.device("hammerblade")
    assert torch.equal(x_h.cpu(), x)

def test_torch_fill_2():
    x = torch.empty(5, 6)
    x_h = x.hammerblade()
    x.fill_(42)
    x_h.fill_(42)
    assert x_h.device == torch.device("hammerblade")
    assert torch.equal(x_h.cpu(), x)

@settings(deadline=None)
@given(tensor=hu.tensor(), scalar=st.floats(width=32))
def test_torch_fill_hypothesis(tensor, scalar):
    assume(scalar != 0)
    assume(not isnan(scalar))
    assume(not isinf(scalar))
    x = torch.tensor(tensor)
    x_h = x.hammerblade()
    x.fill_(scalar)
    x_h.fill_(scalar)
    assert x_h.device == torch.device("hammerblade")
    assert torch.equal(x_h.cpu(), x)
