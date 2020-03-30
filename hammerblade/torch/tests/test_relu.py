"""
Tests on torch.nn.relu (threshold kernel)
03/09/2020 Lin Cheng (lc873@cornell.edu)
03/29/2020 Angela Zou
"""
import torch
import torch.nn as nn
import pytest
from hypothesis import given
import hypothesis.strategies as st
from .hypothesis_test_util import HypothesisUtil as hu

# @pytest.mark.skip(reason="not yet implemented")
def test_torch_nn_relu_1():
    x = torch.ones(10)
    x_h = x.hammerblade()
    relu = nn.ReLU()
    x_relu = relu(x)
    x_h_relu = relu(x_h)
    assert x_h_relu.device == torch.device("hammerblade")
    assert torch.equal(x_h_relu.cpu(), x_relu)

# @pytest.mark.skip(reason="not yet implemented")
def test_torch_nn_relu_2():
    x = torch.randn(10)
    x_h = x.hammerblade()
    relu = nn.ReLU()
    x_relu = relu(x)
    x_h_relu = relu(x_h)
    assert x_h_relu.device == torch.device("hammerblade")
    assert torch.equal(x_h_relu.cpu(), x_relu)

# @pytest.mark.skip(reason="not yet implemented")
def test_torch_nn_relu_3():
    x = torch.randn(3, 4)
    x_h = x.hammerblade()
    relu = nn.ReLU()
    x_relu = relu(x)
    x_h_relu = relu(x_h)
    assert x_h_relu.device == torch.device("hammerblade")
    assert torch.equal(x_h_relu.cpu(), x_relu)

def test_torch_nn_relu_4():
    x = torch.randn(3, 4, 5)
    x_h = x.hammerblade()
    relu = nn.ReLU()
    x_relu = relu(x)
    x_h_relu = relu(x_h)
    assert x_h_relu.device == torch.device("hammerblade")
    assert torch.equal(x_h_relu.cpu(), x_relu)

@given(tensors=hu.tensors(n=2), scalars=st.floats(n=2, min_value=-1024.0, max_value=1024.0, width=32))
def test_elementwise_torch_nn_relu_hypothesis(tensors, scalars):
    def elementwise_torch_nn_relu(tensors, scalars):
        assert len(tensors) == 2
        assert len(scalars) == 2
        tensor_self, tensor_others = tensors
        threshold, value = scalars
        if (tensor_self <= threshold):
            return value
        else:
            return others
    hu.assert_hb_checks(elementwise_torch_nn_relu, [tensors, scalars])



