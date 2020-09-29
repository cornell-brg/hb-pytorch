"""
Unit tests for torch.sqrt kernel
03/06/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import random
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu
from math import isnan

torch.manual_seed(42)
random.seed(42)

def _test_torch_sqrt(x):
    h = x.hammerblade()
    sqrt_x = x.sqrt()
    sqrt_h = h.sqrt()
    assert sqrt_h.device == torch.device("hammerblade")
    assert torch.equal(sqrt_h.cpu(), sqrt_x)

def test_torch_sqrt_1():
    x = torch.ones(10)
    _test_torch_sqrt(x)

def test_torch_sqrt_2():
    x = torch.randn(3, 4)
    x = x.abs()
    _test_torch_sqrt(x)

@settings(deadline=None)
@given(tensor=hu.tensor())
def test_torch_sqrt_hypothesis(tensor):
    x = torch.tensor(tensor)
    x = x.abs()
    _test_torch_sqrt(x)

def test_torch_sqrt_nan():
    x = torch.tensor([-1.])
    h = x.hammerblade()
    sqrt_x = x.sqrt()
    sqrt_h = h.sqrt()
    assert sqrt_h.device == torch.device("hammerblade")
    assert isnan(sqrt_x.item())
    assert isnan(sqrt_h.item())
