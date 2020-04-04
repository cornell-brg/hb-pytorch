"""
BRG tests on PyTorch => tests of real offloading kernels
Feb 09, 2020
Lin Cheng
"""

import torch
import hypothesis.strategies as st
from math import isnan, isinf
from hypothesis import assume, given, settings
from .hypothesis_test_util import HypothesisUtil as hu

torch.manual_seed(42)

def _test_mul(x1, x2):
    h1 = x1.hammerblade()
    h2 = x2.hammerblade()
    assert h1 is not x1
    assert h2 is not x2
    y_c = x1 * x2
    y_h = h1 * h2
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_c, y_h.cpu())
    # inplace
    x1.mul_(x2)
    h1.mul_(h2)
    assert h1.device == torch.device("hammerblade")
    assert torch.allclose(x1, h1.cpu())

def test_mul_1():
    x = torch.ones(1, 10)
    _test_mul(x, x)

def test_mul_2():
    x1 = torch.ones(4, 5)
    x2 = torch.ones(4, 5)
    _test_mul(x1, x2)

def test_mul_3():
    x1 = torch.rand(1, 128)
    x2 = torch.rand(1, 128)
    _test_mul(x1, x2)

def test_mul_4():
    x1 = torch.rand(16, 32)
    x2 = torch.rand(16, 32)
    _test_mul(x1, x2)

@settings(deadline=None)
@given(inputs=hu.tensors(n=2))
def test_mul_hypothesis(inputs):
    x1 = torch.tensor(inputs[0])
    x2 = torch.tensor(inputs[1])
    _test_mul(x1, x2)

def test_mul_with_scalar():
    x = torch.ones(16)
    h = x.hammerblade()
    y_c = x * 5
    y_h = h * 5
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_h.cpu(), y_c)

@settings(deadline=None)
@given(tensor=hu.tensor(), scalar=st.floats(width=32))
def test_mul_with_scalar_hypothesis(tensor, scalar):
    assume(not isnan(scalar))
    assume(not isinf(scalar))
    assume(abs(scalar) > 1e-05)
    x = torch.tensor(tensor)
    h = x.hammerblade()
    y_c = x * scalar
    y_h = h * scalar
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_h.cpu(), y_c)

def _test_div(x1, x2):
    h1 = x1.hammerblade()
    h2 = x2.hammerblade()
    assert h1 is not x1
    assert h2 is not x2
    y_c = x1 / x2
    y_h = h1 / h2
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_c, y_h.cpu())
    # inplace
    x1.div_(x2)
    h1.div_(h2)
    assert h1.device == torch.device("hammerblade")
    assert torch.allclose(x1, h1.cpu())

def test_div_1():
    x = torch.ones(1, 10)
    _test_div(x, x)


def test_div_2():
    x1 = torch.ones(4, 5)
    x2 = torch.ones(4, 5)
    _test_div(x1, x2)

def test_div_3():
    x1 = torch.rand(1, 128)
    x2 = torch.rand(1, 128)
    _test_div(x1, x2)

def test_div_4():
    x1 = torch.rand(16, 32)
    x2 = torch.rand(16, 32)
    _test_div(x1, x2)

@settings(deadline=None)
@given(inputs=hu.tensors(n=2, nonzero=True))
def test_div_hypothesis(inputs):
    x1 = torch.tensor(inputs[0])
    x2 = torch.tensor(inputs[1])
    _test_div(x1, x2)

def test_div_with_scalar():
    x = torch.ones(16)
    h = x.hammerblade()
    y_c = x / 5.0
    y_h = h / 5.0
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_h.cpu(), y_c)

@settings(deadline=None)
@given(tensor=hu.tensor(), scalar=st.floats(width=32))
def test_div_with_scalar_hypothesis(tensor, scalar):
    assume(scalar != 0)
    assume(not isnan(scalar))
    assume(not isinf(scalar))
    assume(abs(scalar) > 1e-05)
    x = torch.tensor(tensor)
    h = x.hammerblade()
    y_c = x / scalar
    y_h = h / scalar
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_h.cpu(), y_c)
