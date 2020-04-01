"""
Tests on torch.nn.CrossEntropyLoss
03/25/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import torch.nn as nn

torch.manual_seed(42)

def _test_torch_nn_CrossEntropyLoss(loss, input, target):
    output = loss(input, target)
    output_h = loss(input.hammerblade(), target.hammerblade())
    assert output_h.device == torch.device("hammerblade")
    assert torch.allclose(output, output_h.cpu())

def test_torch_nn_CrossEntropyLoss_mean():
    loss = nn.CrossEntropyLoss(reduction='mean')
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    _test_torch_nn_CrossEntropyLoss(loss, input, target)

def test_torch_nn_CrossEntropyLoss_none():
    loss = nn.CrossEntropyLoss(reduction='none')
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    _test_torch_nn_CrossEntropyLoss(loss, input, target)

def test_torch_nn_CrossEntropyLoss_sum():
    loss = nn.CrossEntropyLoss(reduction='sum')
    input = torch.randn(3, 5, requires_grad=True)
    target = torch.empty(3, dtype=torch.long).random_(5)
    _test_torch_nn_CrossEntropyLoss(loss, input, target)
