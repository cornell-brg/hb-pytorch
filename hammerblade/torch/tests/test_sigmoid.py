"""
Tests on torch.nn.relu (threshold kernel)
03/09/2020 Lin Cheng (lc873@cornell.edu)
"""
import torch
import torch.nn as nn
import pytest

@pytest.mark.skip(reason="not yet implemented")
def test_torch_nn_sigmoid_1():
    sigmoid = nn.Sigmoid()
    x = torch.ones(10)
    x_h = x.hammerblade()
    x_sig = sigmoid(x)
    x_h_sig = sigmoid(x_h)
    assert x_h_sig.device == torch.device("hammerblade")
    assert torch.allclose(x_h_sig.cpu(), x_sig)

@pytest.mark.skip(reason="not yet implemented")
def test_torch_nn_sigmoid_2():
    sigmoid = nn.Sigmoid()
    x = torch.randn(10)
    x_h = x.hammerblade()
    x_sig = sigmoid(x)
    x_h_sig = sigmoid(x_h)
    assert x_h_sig.device == torch.device("hammerblade")
    assert torch.allclose(x_h_sig.cpu(), x_sig)

@pytest.mark.skip(reason="not yet implemented")
def test_torch_nn_sigmoid_3():
    sigmoid = nn.Sigmoid()
    x = torch.randn(3, 4)
    x_h = x.hammerblade()
    x_sig = sigmoid(x)
    x_h_sig = sigmoid(x_h)
    assert x_h_sig.device == torch.device("hammerblade")
    assert torch.allclose(x_h_sig.cpu(), x_sig)
