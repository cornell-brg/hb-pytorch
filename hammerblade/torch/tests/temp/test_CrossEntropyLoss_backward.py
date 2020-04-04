"""
Tests on torch.nn.CrossEntropyLoss backward
03/25/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import torch.nn as nn
import hbutils

torch.manual_seed(42)

def _test_torch_nn_CrossEntropyLoss_back(loss, input, target):
    input_h = hbutils.init_hb_tensor(input)
    assert input_h is not input
    output = loss(input, target)
    output_h = loss(input_h, target.hammerblade())
    assert output_h.device == torch.device("hammerblade")
    assert torch.allclose(output, output_h.cpu())
    output.backward()
    output_h.backward()
    assert input.grad is not None
    assert input_h.grad is not None
    assert input.grad is not input_h.grad
    assert torch.allclose(input.grad, input_h.grad.cpu())

def test_torch_nn_CrossEntropyLoss_mean_back():
    loss = nn.CrossEntropyLoss(reduction='mean')
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    _test_torch_nn_CrossEntropyLoss_back(loss, input, target)

def test_torch_nn_CrossEntropyLoss_sum_back():
    loss = nn.CrossEntropyLoss(reduction='sum')
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    _test_torch_nn_CrossEntropyLoss_back(loss, input, target)

def test_torch_nn_CrossEntropyLoss_none_back():
    loss = nn.CrossEntropyLoss(reduction='none')
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    grad = torch.tensor([1., 2., 3.])
    input_h = hbutils.init_hb_tensor(input)
    assert input_h is not input
    output = loss(input, target)
    output_h = loss(input_h, target.hammerblade())
    assert output_h.device == torch.device("hammerblade")
    assert torch.allclose(output, output_h.cpu())
    output.backward(grad)
    output_h.backward(grad.hammerblade())
    assert input.grad is not None
    assert input_h.grad is not None
    assert input.grad is not input_h.grad
    assert torch.allclose(input.grad, input_h.grad.cpu())
