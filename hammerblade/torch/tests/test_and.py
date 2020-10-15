"""
tests of and kernel
Authors : Janice Wei
Date    : 09/25/2020
"""

import torch
import random
from hypothesis import given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

# ------------------------------------------------------------------------
# test of x1 & x2
# ------------------------------------------------------------------------

def _test_and(x1, x2):
    h1 = x1.hammerblade()
    h2 = x2.hammerblade()
    assert h1 is not x1
    assert h2 is not x2
    y_c = x1 & x2
    y_h = h1 & h2
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_c, y_h.cpu())

# ------------------------------------------------------------------------
# tests of and kernel with integer elements
# ------------------------------------------------------------------------

def test_and_1():
    x = torch.ones(1, 10, dtype=torch.int)
    _test_and(x, x)

def test_and_2():
    x1 = torch.ones(4, 5, dtype=torch.int)
    x2 = torch.ones(4, 5, dtype=torch.int)
    _test_and(x1, x2)

def test_and_3():
    x = torch.randint(-2 ** 30, 2 ** 30 - 1, (1, 128)).to(torch.int32)
    y = torch.randint(-2 ** 30, 2 ** 30 - 1, (1, 128)).to(torch.int32)
    _test_and(x, y)

def test_and_4():
    x = torch.randint(-2 ** 30, 2 ** 30 - 1, (16, 32)).to(torch.int32)
    y = torch.randint(-2 ** 30, 2 ** 30 - 1, (16, 32)).to(torch.int32)
    _test_and(x, y)

def test_and_bool1():
    x = torch.randint(0, 2, (16, 32)).to(torch.bool)
    y = torch.randint(0, 2, (16, 32)).to(torch.bool)
    _test_and(x, y)

def test_and_bool2():
    x = torch.randint(0, 2, (1, 128)).to(torch.bool)
    y = torch.randint(0, 2, (1, 128)).to(torch.bool)
    _test_and(x, y)

@settings(deadline=None)
@given(inputs=hu.tensors(n=2))
def test_and_hypothesis(inputs):
    x1 = torch.tensor(inputs[0]).to(torch.int32)
    x2 = torch.tensor(inputs[1]).to(torch.int32)
    _test_and(x1, x2)
