"""
tests of lt, gt kernel with float, int, and long long elements.
Authors : Zhongyuan Zhao
Date    : 01/27/2022
"""

import torch
import random
from math import isnan, isinf
from hypothesis import assume, given, settings
import hypothesis.strategies as st
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)
random.seed(42)

# ------------------------------------------------------------------------
# test of lting two tensors x1 and x2
# ------------------------------------------------------------------------

def _test_lt(x1, x2):
    h1 = x1.hammerblade()
    h2 = x2.hammerblade()
    assert h1 is not x1
    assert h2 is not x2
    y_c = torch.lt(x1, x2)
    y_h = torch.lt(h1, h2)
    assert y_h.device == torch.device("hammerblade")
    assert torch.equal(y_c, y_h.cpu())

# ------------------------------------------------------------------------
# tests of lt kernel with float elements
# ------------------------------------------------------------------------

def test_lt_1():
    x = torch.ones(1, 10)
    _test_lt(x, x)

def test_lt_2():
    x1 = torch.ones(4, 5)
    x2 = torch.ones(4, 5)
    _test_lt(x1, x2)

def test_lt_3():
    x1 = torch.rand(1, 128)
    x2 = torch.rand(1, 128)
    _test_lt(x1, x2)

def test_lt_4():
    x1 = torch.rand(16, 32)
    x2 = torch.rand(16, 32)
    _test_lt(x1, x2)

@settings(deadline=None)
@given(inputs=hu.tensors(n=2))
def test_lt_hypothesis(inputs):
    x1 = torch.tensor(inputs[0])
    x2 = torch.tensor(inputs[1])
    x1 = x1.to(torch.float32)
    x2 = x2.to(torch.float32)
    _test_lt(x1, x2)

def test_lt_with_scalar():
    x = torch.ones(16)
    h = x.hammerblade()
    y_c = torch.lt(x, 5)
    y_h = torch.lt(h, 5)
    assert y_h.device == torch.device("hammerblade")
    assert torch.equal(y_h.cpu(), y_c)

@settings(deadline=None)
@given(tensor=hu.tensor(), scalar=st.floats(width=32))
def test_lt_with_scalar_hypothesis(tensor, scalar):
    assume(not isnan(scalar))
    assume(not isinf(scalar))
    assume(abs(scalar) > 1e-05)
    x = torch.tensor(tensor)
    h = x.hammerblade()
    y_c = torch.lt(x, scalar)
    y_h = torch.lt(h, scalar)
    assert y_h.device == torch.device("hammerblade")
    assert torch.equal(y_h.cpu(), y_c)

# ------------------------------------------------------------------------
# tests of lt kernel with int (32bit) elements
# ------------------------------------------------------------------------

def test_lt_int_1():
    x = torch.ones(1, 10, dtype=torch.int)
    _test_lt(x, x)

def test_lt_int_2():
    x1 = torch.ones(4, 5, dtype=torch.int)
    x2 = torch.ones(4, 5, dtype=torch.int)
    _test_lt(x1, x2)

def test_lt_int_3():
    x1 = torch.randint(-2 ** 30, 2 ** 30 - 1, (1, 128))
    x = x1.to(torch.int32)
    x2 = torch.randint(-2 ** 30, 2 ** 30 - 1, (1, 128))
    y = x2.to(torch.int32)
    _test_lt(x, y)

def test_lt_int_4():
    x1 = torch.randint(-2 ** 30, 2 ** 30 - 1, (16, 32))
    x = x1.to(torch.int32)
    x2 = torch.randint(-2 ** 30, 2 ** 30 - 1, (16, 32))
    y = x2.to(torch.int32)
    _test_lt(x, y)

@settings(deadline=None)
@given(inputs=hu.tensors(n=2))
def test_lt_int_hypothesis(inputs):
    x1 = torch.tensor(inputs[0])
    x2 = torch.tensor(inputs[1])
    x1 = x1.to(torch.int32)
    x2 = x2.to(torch.int32)
    _test_lt(x1, x2)

def test_lt_int_with_scalar():
    x = torch.ones(16, dtype=torch.int)
    h = x.hammerblade()
    y_c = torch.lt(x, 5)
    y_h = torch.lt(h, 5)
    assert y_h.device == torch.device("hammerblade")
    assert torch.equal(y_h.cpu(), y_c)

# ------------------------------------------------------------------------
# test of gttracting x2 from x1
# ------------------------------------------------------------------------

def _test_gt(x1, x2):
    h1 = x1.hammerblade()
    h2 = x2.hammerblade()
    assert h1 is not x1
    assert h2 is not x2
    y_c = torch.gt(x1, x2)
    y_h = torch.gt(h1, h2)
    assert y_h.device == torch.device("hammerblade")
    assert torch.equal(y_c, y_h.cpu())

# ------------------------------------------------------------------------
# tests of gt kernel with float elements
# ------------------------------------------------------------------------

def test_gt_1():
    x = torch.ones(1, 10)
    _test_gt(x, x)

def test_gt_2():
    x1 = torch.ones(4, 5)
    x2 = torch.ones(4, 5)
    _test_gt(x1, x2)

def test_gt_3():
    x1 = torch.rand(1, 128)
    x2 = torch.rand(1, 128)
    _test_gt(x1, x2)

def test_gt_4():
    x1 = torch.rand(16, 32)
    x2 = torch.rand(16, 32)
    _test_gt(x1, x2)

@settings(deadline=None)
@given(inputs=hu.tensors(n=2))
def test_gt_hypothesis(inputs):
    x1 = torch.tensor(inputs[0])
    x2 = torch.tensor(inputs[1])
    x1 = x1.to(torch.float32)
    x2 = x2.to(torch.float32)
    _test_gt(x1, x2)

def test_gt_with_scalar():
    x = torch.ones(16)
    h = x.hammerblade()
    y_c = torch.gt(x, 5)
    y_h = torch.gt(h, 5)
    assert y_h.device == torch.device("hammerblade")
    assert torch.equal(y_h.cpu(), y_c)

@settings(deadline=None)
@given(tensor=hu.tensor(), scalar=st.floats(width=32))
def test_gt_with_scalar_hypothesis(tensor, scalar):
    assume(not isnan(scalar))
    assume(not isinf(scalar))
    assume(abs(scalar) > 1e-05)
    x = torch.tensor(tensor)
    h = x.hammerblade()
    y_c = torch.gt(x, scalar)
    y_h = torch.gt(h, scalar)
    assert y_h.device == torch.device("hammerblade")
    assert torch.equal(y_h.cpu(), y_c)

# ------------------------------------------------------------------------
# tests of gt kernel with int (32bit) elements
# ------------------------------------------------------------------------

def test_gt_int_1():
    x = torch.ones(1, 10, dtype=torch.int)
    _test_gt(x, x)

def test_gt_int_2():
    x1 = torch.ones(4, 5, dtype=torch.int)
    x2 = torch.ones(4, 5, dtype=torch.int)
    _test_gt(x1, x2)

def test_gt_int_3():
    x1 = torch.randint(-2 ** 30, 2 ** 30 - 1, (1, 128))
    x = x1.to(torch.int32)
    x2 = torch.randint(-2 ** 30, 2 ** 30 - 1, (1, 128))
    y = x2.to(torch.int32)
    _test_gt(x, y)

def test_gt_int_4():
    x1 = torch.randint(-2 ** 30, 2 ** 30 - 1, (16, 32))
    x = x1.to(torch.int32)
    x2 = torch.randint(-2 ** 30, 2 ** 30 - 1, (16, 32))
    y = x2.to(torch.int32)
    _test_gt(x, y)

@settings(deadline=None)
@given(inputs=hu.tensors(n=2))
def test_gt_int_hypothesis(inputs):
    x1 = torch.tensor(inputs[0])
    x2 = torch.tensor(inputs[1])
    x1 = x1.to(torch.int32)
    x2 = x2.to(torch.int32)
    _test_gt(x1, x2)

def test_gt_int_with_scalar():
    x = torch.ones(16, dtype=torch.int)
    h = x.hammerblade()
    y_c = torch.gt(x, 5)
    y_h = torch.gt(h, 5)
    assert y_h.device == torch.device("hammerblade")
    assert torch.equal(y_h.cpu(), y_c)

