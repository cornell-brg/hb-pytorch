"""
Tests for log
06/16/2020 Lin Cheng (lc873@cornell.edu)
"""
import torch
import random
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

def _test_torch_log(x):
    h = x.hammerblade()
    log_x = x.log()
    log_h = h.log()
    assert log_h.device == torch.device("hammerblade")
    assert torch.allclose(log_h.cpu(), log_x, equal_nan=True)

def test_torch_log_1():
    x = torch.randn(3, 5)
    _test_torch_log(x)

def test_torch_log_2():
    x = torch.ones(10)
    _test_torch_log(x)

def test_torch_log_3():
    x = torch.zeros(5)
    _test_torch_log(x)

@settings(deadline=None)
@given(tensor=hu.tensor())
def test_torch_log_hypothesis(tensor):
    x = torch.tensor(tensor)
    _test_torch_log(x)
