"""
tests of add and sub kernel with float, int, and long long elements.
Authors : Lin Cheng, Janice Wei
Date    : 02/09/2020, 07/13/2020
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
# test of adding two tensors x1 and x2
# ------------------------------------------------------------------------

def _test_add(x1, x2):
    h1 = x1.hammerblade()
    h2 = x2.hammerblade()
    assert h1 is not x1
    assert h2 is not x2
    y_c = x1 + x2
    y_h = h1 + h2
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_c, y_h.cpu())
    # inplace
    x1.add_(x2)
    h1.add_(h2)
    assert h1.device == torch.device("hammerblade")
    assert torch.allclose(x1, h1.cpu())

# ------------------------------------------------------------------------
# tests of add kernel with float elements
# ------------------------------------------------------------------------

def test_add_1():
    x = torch.ones(1, 10)
    _test_add(x, x)

def test_add_2():
    x1 = torch.ones(4, 5)
    x2 = torch.ones(4, 5)
    _test_add(x1, x2)

def test_add_3():
    x1 = torch.rand(1, 128)
    x2 = torch.rand(1, 128)
    _test_add(x1, x2)

def test_add_4():
    x1 = torch.rand(16, 32)
    x2 = torch.rand(16, 32)
    _test_add(x1, x2)

@settings(deadline=None)
@given(inputs=hu.tensors(n=2))
def test_add_hypothesis(inputs):
    x1 = torch.tensor(inputs[0])
    x2 = torch.tensor(inputs[1])
    x1 = x1.to(torch.float32)
    x2 = x2.to(torch.float32)
    _test_add(x1, x2)

def test_add_with_scalar():
    x = torch.ones(16)
    h = x.hammerblade()
    y_c = x + 5
    y_h = h + 5
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_h.cpu(), y_c)

@settings(deadline=None)
@given(tensor=hu.tensor(), scalar=st.floats(width=32))
def test_add_with_scalar_hypothesis(tensor, scalar):
    assume(not isnan(scalar))
    assume(not isinf(scalar))
    assume(abs(scalar) > 1e-05)
    x = torch.tensor(tensor)
    h = x.hammerblade()
    y_c = x + scalar
    y_h = h + scalar
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_h.cpu(), y_c)

# ------------------------------------------------------------------------
# tests of add kernel with int (32bit) elements
# ------------------------------------------------------------------------

def test_add_int_1():
    x = torch.ones(1, 10, dtype=torch.int)
    _test_add(x, x)

def test_add_int_2():
    x1 = torch.ones(4, 5, dtype=torch.int)
    x2 = torch.ones(4, 5, dtype=torch.int)
    _test_add(x1, x2)

def test_add_int_3():
    x1 = torch.randint(-2 ** 30, 2 ** 30 - 1, (1, 128))
    x = x1.to(torch.int32)
    x2 = torch.randint(-2 ** 30, 2 ** 30 - 1, (1, 128))
    y = x2.to(torch.int32)
    _test_add(x, y)

def test_add_int_4():
    x1 = torch.randint(-2 ** 30, 2 ** 30 - 1, (16, 32))
    x = x1.to(torch.int32)
    x2 = torch.randint(-2 ** 30, 2 ** 30 - 1, (16, 32))
    y = x2.to(torch.int32)
    _test_add(x, y)

@settings(deadline=None)
@given(inputs=hu.tensors(n=2))
def test_add_int_hypothesis(inputs):
    x1 = torch.tensor(inputs[0])
    x2 = torch.tensor(inputs[1])
    x1 = x1.to(torch.int32)
    x2 = x2.to(torch.int32)
    _test_add(x1, x2)

def test_add_int_with_scalar():
    x = torch.ones(16, dtype=torch.int)
    h = x.hammerblade()
    y_c = x + 5
    y_h = h + 5
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_h.cpu(), y_c)

# ------------------------------------------------------------------------
# tests of add kernel with long long (64bit) elements
# ------------------------------------------------------------------------

def test_add_long_1():
    x = torch.ones(1, 10, dtype=torch.long)
    _test_add(x, x)

def test_add_long_2():
    x1 = torch.ones(4, 5, dtype=torch.long)
    x2 = torch.ones(4, 5, dtype=torch.long)
    _test_add(x1, x2)

def test_add_long_3():
    x1 = torch.randint(-2 ** 62 + 1, 2 ** 62 - 1, (1, 128))
    x2 = torch.randint(-2 ** 62 + 1, 2 ** 62 - 1, (1, 128))
    _test_add(x1, x2)

def test_add_long_4():
    x1 = torch.randint(-2 ** 62 + 1, 2 ** 62 - 1, (16, 32))
    x2 = torch.randint(-2 ** 62 + 1, 2 ** 62 - 1, (16, 32))
    _test_add(x1, x2)

@settings(deadline=None)
@given(inputs=hu.tensors(n=2))
def test_add_long_hypothesis(inputs):
    x1 = torch.tensor(inputs[0])
    x2 = torch.tensor(inputs[1])
    _test_add(x1, x2)

def test_add_long_with_scalar():
    x = torch.ones(16, dtype=torch.long)
    h = x.hammerblade()
    y_c = x + 5
    y_h = h + 5
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_h.cpu(), y_c)

# ------------------------------------------------------------------------
# test of subtracting x2 from x1
# ------------------------------------------------------------------------

def _test_sub(x1, x2):
    h1 = x1.hammerblade()
    h2 = x2.hammerblade()
    assert h1 is not x1
    assert h2 is not x2
    y_c = x1 - x2
    y_h = h1 - h2
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_c, y_h.cpu())
    # inplace
    x1.sub_(x2)
    h1.sub_(h2)
    assert h1.device == torch.device("hammerblade")
    assert torch.allclose(x1, h1.cpu())

# ------------------------------------------------------------------------
# tests of sub kernel with float elements
# ------------------------------------------------------------------------

def test_sub_1():
    x = torch.ones(1, 10)
    _test_sub(x, x)

def test_sub_2():
    x1 = torch.ones(4, 5)
    x2 = torch.ones(4, 5)
    _test_sub(x1, x2)

def test_sub_3():
    x1 = torch.rand(1, 128)
    x2 = torch.rand(1, 128)
    _test_sub(x1, x2)

def test_sub_4():
    x1 = torch.rand(16, 32)
    x2 = torch.rand(16, 32)
    _test_sub(x1, x2)

@settings(deadline=None)
@given(inputs=hu.tensors(n=2))
def test_sub_hypothesis(inputs):
    x1 = torch.tensor(inputs[0])
    x2 = torch.tensor(inputs[1])
    x1 = x1.to(torch.float32)
    x2 = x2.to(torch.float32)
    _test_sub(x1, x2)

def test_sub_with_scalar():
    x = torch.ones(16)
    h = x.hammerblade()
    y_c = x - 5
    y_h = h - 5
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_h.cpu(), y_c)

@settings(deadline=None)
@given(tensor=hu.tensor(), scalar=st.floats(width=32))
def test_sub_with_scalar_hypothesis(tensor, scalar):
    assume(not isnan(scalar))
    assume(not isinf(scalar))
    assume(abs(scalar) > 1e-05)
    x = torch.tensor(tensor)
    h = x.hammerblade()
    y_c = x - scalar
    y_h = h - scalar
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_h.cpu(), y_c)

# ------------------------------------------------------------------------
# tests of sub kernel with int (32bit) elements
# ------------------------------------------------------------------------

def test_sub_int_1():
    x = torch.ones(1, 10, dtype=torch.int)
    _test_sub(x, x)

def test_sub_int_2():
    x1 = torch.ones(4, 5, dtype=torch.int)
    x2 = torch.ones(4, 5, dtype=torch.int)
    _test_sub(x1, x2)

def test_sub_int_3():
    x1 = torch.randint(-2 ** 30, 2 ** 30 - 1, (1, 128))
    x = x1.to(torch.int32)
    x2 = torch.randint(-2 ** 30, 2 ** 30 - 1, (1, 128))
    y = x2.to(torch.int32)
    _test_sub(x, y)

def test_sub_int_4():
    x1 = torch.randint(-2 ** 30, 2 ** 30 - 1, (16, 32))
    x = x1.to(torch.int32)
    x2 = torch.randint(-2 ** 30, 2 ** 30 - 1, (16, 32))
    y = x2.to(torch.int32)
    _test_sub(x, y)

@settings(deadline=None)
@given(inputs=hu.tensors(n=2))
def test_sub_int_hypothesis(inputs):
    x1 = torch.tensor(inputs[0])
    x2 = torch.tensor(inputs[1])
    x1 = x1.to(torch.int32)
    x2 = x2.to(torch.int32)
    _test_sub(x1, x2)

def test_sub_int_with_scalar():
    x = torch.ones(16, dtype=torch.int)
    h = x.hammerblade()
    y_c = x - 5
    y_h = h - 5
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_h.cpu(), y_c)

# ------------------------------------------------------------------------
# tests of sub kernel with long long (64bit) elements
# ------------------------------------------------------------------------

def test_sub_long_1():
    x = torch.ones(1, 10, dtype=torch.long)
    _test_sub(x, x)

def test_sub_long_2():
    x1 = torch.ones(4, 5, dtype=torch.long)
    x2 = torch.ones(4, 5, dtype=torch.long)
    _test_sub(x1, x2)

def test_sub_long_3():
    x1 = torch.randint(-2 ** 62 + 1, 2 ** 62 - 1, (1, 128))
    x2 = torch.randint(-2 ** 62 + 1, 2 ** 62 - 1, (1, 128))
    _test_sub(x1, x2)

def test_sub_long_4():
    x1 = torch.randint(-2 ** 62 + 1, 2 ** 62 - 1, (16, 32))
    x2 = torch.randint(-2 ** 62 + 1, 2 ** 62 - 1, (16, 32))
    _test_sub(x1, x2)

@settings(deadline=None)
@given(inputs=hu.tensors(n=2))
def test_sub_long_hypothesis(inputs):
    x1 = torch.tensor(inputs[0])
    x2 = torch.tensor(inputs[1])
    _test_sub(x1, x2)

def test_sub_long_with_scalar():
    x = torch.ones(16, dtype=torch.long)
    h = x.hammerblade()
    y_c = x - 5
    y_h = h - 5
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_h.cpu(), y_c)
