"""
BRG tests on PyTorch => tests of real offloading kernels
Feb 09, 2020
Lin Cheng
"""

import torch
import hypothesis.strategies as st
from math import isnan, isinf
from hypothesis import assume, given
from .hypothesis_test_util import HypothesisUtil as hu

def test_elementwise_mul_1():
    x = torch.ones(1, 10)
    y = torch.ones(1, 10)
    z = x * y
    z_h = x.hammerblade() * y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.equal(z_h.cpu(), z)

def test_elementwise_mul_2():
    x = torch.ones(4, 5)
    y = torch.ones(4, 5)
    z = x * y
    z_h = x.hammerblade() * y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.equal(z_h.cpu(), z)

def test_elementwise_mul_3():
    x = torch.rand(1, 128)
    y = torch.rand(1, 128)
    z = x * y
    z_h = x.hammerblade() * y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.equal(z_h.cpu(), z)

def test_elementwise_mul_4():
    x = torch.rand(16, 32)
    y = torch.rand(16, 32)
    z = x * y
    z_h = x.hammerblade() * y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.equal(z_h.cpu(), z)

@given(inputs=hu.tensors(n=2))
def test_elementwise_mul_hypothesis(inputs):
    def elementwise_mul(inputs):
        assert len(inputs) == 2
        return inputs[0] * inputs[1]
    hu.assert_hb_checks(elementwise_mul, inputs)

def test_elementwise_in_place_mul():
    x1 = torch.rand(16, 32)
    x2 = torch.rand(16, 32)
    x1_h = x1.hammerblade()
    x2_h = x2.hammerblade()
    x1.mul_(x2)
    x1_h.mul_(x2_h)
    assert x1_h.device == torch.device("hammerblade")
    x1_h_c = x1_h.cpu()
    assert torch.equal(x1_h_c, x1)

@given(inputs=hu.tensors(n=2))
def test_elementwise_in_place_mul_hypothesis(inputs):
    def elementwise_mul(inputs):
        x1, x2 = inputs
        x1.mul_(x2)
        return x1
    hu.assert_hb_checks(elementwise_mul, inputs)

def test_mul_with_scalar():
    x = torch.rand(16)
    x_h = x.hammerblade()
    y = x.mul(42.0)
    y_h = x_h.mul(42.0)
    assert y_h.device == torch.device("hammerblade")
    assert torch.equal(y_h.cpu(), y)

@given(tensor=hu.tensor(), scalar=st.floats(width=32))
def test_mul_with_scalar_hypothesis(tensor, scalar):
    assume(not isnan(scalar))
    assume(not isinf(scalar))
    def mul_scalar(inputs):
        tensor, scalar = inputs
        return tensor * scalar
    hu.assert_hb_checks(mul_scalar, [tensor, scalar])

def test_elementwise_div_1():
    x = torch.ones(1, 10)
    y = torch.ones(1, 10)
    z = x / y
    z_h = x.hammerblade() / y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.allclose(z_h.cpu(), z)

def test_elementwise_div_2():
    x = torch.ones(4, 5)
    y = torch.ones(4, 5)
    z = x / y
    z_h = x.hammerblade() / y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.allclose(z_h.cpu(), z)

def test_elementwise_div_3():
    x = torch.rand(1, 128)
    y = torch.rand(1, 128)
    z = x / y
    z_h = x.hammerblade() / y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.allclose(z_h.cpu(), z)

def test_elementwise_div_4():
    x = torch.rand(16, 32)
    y = torch.rand(16, 32)
    z = x / y
    z_h = x.hammerblade() / y.hammerblade()
    assert z_h.device == torch.device("hammerblade")
    assert torch.allclose(z_h.cpu(), z)

@given(inputs=hu.tensors(n=2, nonzero=True))
def test_elementwise_div_hypothesis(inputs):
    def elementwise_div(inputs):
        assert len(inputs) == 2
        return inputs[0] / inputs[1]
    hu.assert_hb_checks(elementwise_div, inputs)

def test_elementwise_in_place_div():
    x1 = torch.rand(16, 32)
    x2 = torch.rand(16, 32)
    x1_h = x1.hammerblade()
    x2_h = x2.hammerblade()
    x1.div_(x2)
    x1_h.div_(x2_h)
    assert x1_h.device == torch.device("hammerblade")
    x1_h_c = x1_h.cpu()
    assert torch.allclose(x1_h_c, x1)

@given(inputs=hu.tensors(n=2, nonzero=True))
def test_elementwise_in_place_div_hypothesis(inputs):
    def elementwise_div(inputs):
        x1, x2 = inputs
        x1.div_(x2)
        return x1
    hu.assert_hb_checks(elementwise_div, inputs)

def test_div_with_scalar():
    x = torch.rand(16)
    x_h = x.hammerblade()
    y = x.div(42.0)
    y_h = x_h.div(42.0)
    assert y_h.device == torch.device("hammerblade")
    assert torch.allclose(y_h.cpu(), y)

@given(tensor=hu.tensor(), scalar=st.floats(width=32))
def test_div_with_scalar_hypothesis(tensor, scalar):
    assume(scalar != 0)
    assume(not isnan(scalar))
    assume(not isinf(scalar))
    def div_scalar(inputs):
        tensor, scalar = inputs
        return tensor / scalar
    hu.assert_hb_checks(div_scalar, [tensor, scalar])
