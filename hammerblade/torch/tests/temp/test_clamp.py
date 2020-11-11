"""
Unit tests for torch.clamp
04/23/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import random
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_torch_clamp(x):
    x = x.clone()
    h = x.clone().hammerblade()
    x_clamp = torch.clamp(x, min=-0.5, max=0.5)
    h_clamp = torch.clamp(h, min=-0.5, max=0.5)
    assert h_clamp.device == torch.device("hammerblade")
    assert torch.equal(h_clamp.cpu(), x_clamp)

def _test_torch_clamp_min(x):
    h = x.hammerblade()
    x_clamp = torch.clamp(x, min=-0.5)
    h_clamp = torch.clamp(h, min=-0.5)
    assert h_clamp.device == torch.device("hammerblade")
    assert torch.equal(h_clamp.cpu(), x_clamp)

def _test_torch_clamp_max(x):
    h = x.hammerblade()
    x_clamp = torch.clamp(x, max=0.5)
    h_clamp = torch.clamp(h, max=0.5)
    assert h_clamp.device == torch.device("hammerblade")
    assert torch.equal(h_clamp.cpu(), x_clamp)

def test_torch_clamp_1():
    x = torch.ones(10)
    _test_torch_clamp(x)
    _test_torch_clamp_min(x)
    _test_torch_clamp_max(x)

def test_torch_clamp_2():
    x = torch.randn(3, 4)
    _test_torch_clamp(x)
    _test_torch_clamp_min(x)
    _test_torch_clamp_max(x)

@settings(deadline=None)
@given(tensor=hu.tensor())
def test_torch_clamp_hypothesis(tensor):
    x = torch.tensor(tensor)
    _test_torch_clamp(x)
    _test_torch_clamp_min(x)
    _test_torch_clamp_max(x)
