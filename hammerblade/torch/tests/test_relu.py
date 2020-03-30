"""
Tests on torch.nn.relu (threshold kernel)
03/09/2020 Lin Cheng (lc873@cornell.edu)
03/29/2020 Angela Zou (az292@cornell.edu)
"""

import torch
import torch.nn as nn
from hypothesis import given
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

def _test_torch_relu_check(tensor_self):
    tensor_self_hb = torch.tensor(tensor_self).hammerblade()
    result_hb = torch.relu(tensor_self_hb)
    assert result_hb.device == torch.device("hammerblade")
    assert torch.allclose(result_hb.cpu(), torch.relu(torch.tensor(tensor_self)))

@given(tensor=hu.tensor())
def test_elementwise_torch_nn_relu_hypothesis(tensor):
    _test_torch_relu_check(tensor)
