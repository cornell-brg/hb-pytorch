"""
tests of not kernel
Authors : Janice Wei
Date    : 10/08/2020
"""

import torch
import random
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

# ------------------------------------------------------------------------
# test of ~x
# ------------------------------------------------------------------------

def _test_not(x):
    h = x.hammerblade()
    assert h is not x
    y_c = ~x
    y_h = ~h
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_c, y_h.cpu())

# ------------------------------------------------------------------------
# tests of not kernel with integer elements
# ------------------------------------------------------------------------

def test_not_1():
    x = torch.ones(1, 10, dtype=torch.int)
    _test_not(x)

def test_not_2():
    x = torch.ones(4, 5, dtype=torch.int)
    _test_not(x)

def test_not_3():
    x = torch.randint(-2 ** 30, 2 ** 30 - 1, (1, 128)).to(torch.int32)
    _test_not(x)

def test_not_4():
    x = torch.randint(-2 ** 30, 2 ** 30 - 1, (16, 32)).to(torch.int32)
    _test_not(x)

@settings(deadline=None)
@given(inputs=hu.tensors(n=1))
def test_or_hypothesis(inputs):
    x = torch.tensor(inputs[0]).to(torch.int32)
    _test_not(x)