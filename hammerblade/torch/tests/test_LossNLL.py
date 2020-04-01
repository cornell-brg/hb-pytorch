"""
Tests on torch.nn.NLLLoss
03/25/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import torch.nn as nn

torch.manual_seed(42)

def _test_torch_nn_NLLLoss(loss, input, target):
    m = nn.LogSoftmax(dim=1)
    output = loss(m(input), target)
    output_h = loss(m(input.hammerblade()), target.hammerblade())
    assert output_h.device == torch.device("hammerblade")
    assert torch.allclose(output, output_h.cpu())

def test_torch_nn_NLLLoss_mean():
    loss = nn.NLLLoss(reduction='mean')
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    _test_torch_nn_NLLLoss(loss, input, target)

def test_torch_nn_NLLLoss_none():
    loss = nn.NLLLoss(reduction='none')
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    _test_torch_nn_NLLLoss(loss, input, target)

def test_torch_nn_NLLLoss_sum():
    loss = nn.NLLLoss(reduction='sum')
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    _test_torch_nn_NLLLoss(loss, input, target)

def _test_torch_nn_NLLLoss_weight(loss, loss_h, input, target):
    m = nn.LogSoftmax(dim=1)
    output = loss(m(input), target)
    output_h = loss_h(m(input.hammerblade()), target.hammerblade())
    assert output_h.device == torch.device("hammerblade")
    assert torch.allclose(output, output_h.cpu())

def test_torch_nn_NLLLoss_mean_weight():
    loss = nn.NLLLoss(reduction='mean', weight=torch.tensor([1., 2., 3., 4., 5.]))
    loss_h = nn.NLLLoss(reduction='mean', weight=torch.tensor([1., 2., 3., 4., 5.]).hammerblade())
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    _test_torch_nn_NLLLoss_weight(loss, loss_h, input, target)

def test_torch_nn_NLLLoss_none_weight():
    loss = nn.NLLLoss(reduction='none', weight=torch.tensor([1., 2., 3., 4., 5.]))
    loss_h = nn.NLLLoss(reduction='none', weight=torch.tensor([1., 2., 3., 4., 5.]).hammerblade())
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    _test_torch_nn_NLLLoss_weight(loss, loss_h, input, target)

def test_torch_nn_NLLLoss_sum_weight():
    loss = nn.NLLLoss(reduction='sum', weight=torch.tensor([1., 2., 3., 4., 5.]))
    loss_h = nn.NLLLoss(reduction='sum', weight=torch.tensor([1., 2., 3., 4., 5.]).hammerblade())
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    _test_torch_nn_NLLLoss_weight(loss, loss_h, input, target)
