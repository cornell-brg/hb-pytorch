"""
Tests on torch.nn.NLLLoss backward
03/26/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import torch.nn as nn
import numpy as np

def test_torch_nn_NLLLoss_mean_back_1():
    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss(reduction='mean')
    input = torch.tensor([[-3.5164, -1.9673, -4.7514, -0.2075, -4.6912],
                         [-2.9321, -1.5085, -0.8699, -1.4016, -2.8089],
                         [-2.6460, -1.9800, -1.4818, -1.8358, -0.9057]],
                         requires_grad=True)
    input_h = torch.tensor([[-3.5164, -1.9673, -4.7514, -0.2075, -4.6912],
                           [-2.9321, -1.5085, -0.8699, -1.4016, -2.8089],
                           [-2.6460, -1.9800, -1.4818, -1.8358, -0.9057]],
                           requires_grad=True)
    target = torch.tensor([1, 0, 4])
    output = loss(m(input), target)
    output_h = loss(m(input_h).hammerblade(), target.hammerblade())
    assert output_h.device == torch.device("hammerblade")
    assert torch.allclose(output, output_h.cpu())
    output.backward()
    output_h.backward()
    assert input.grad is not None
    assert input_h.grad is not None
    assert torch.allclose(input.grad, input_h.grad)

def test_torch_nn_NLLLoss_mean_back_2():
    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss(reduction='mean')
    init = np.random.rand(3, 5)
    input = torch.tensor(init, dtype=torch.float, requires_grad=True)
    input_h = torch.tensor(init, dtype=torch.float, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    output = loss(m(input), target)
    output_h = loss(m(input_h).hammerblade(), target.hammerblade())
    assert output_h.device == torch.device("hammerblade")
    assert torch.allclose(output, output_h.cpu())
    output.backward()
    output_h.backward()
    assert input.grad is not None
    assert input_h.grad is not None
    assert torch.allclose(input.grad, input_h.grad)

def test_torch_nn_NLLLoss_sum_back_1():
    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss(reduction='sum')
    input = torch.tensor([[-3.5164, -1.9673, -4.7514, -0.2075, -4.6912],
                         [-2.9321, -1.5085, -0.8699, -1.4016, -2.8089],
                         [-2.6460, -1.9800, -1.4818, -1.8358, -0.9057]],
                         requires_grad=True)
    input_h = torch.tensor([[-3.5164, -1.9673, -4.7514, -0.2075, -4.6912],
                           [-2.9321, -1.5085, -0.8699, -1.4016, -2.8089],
                           [-2.6460, -1.9800, -1.4818, -1.8358, -0.9057]],
                           requires_grad=True)
    target = torch.tensor([1, 0, 4])
    output = loss(m(input), target)
    output_h = loss(m(input_h).hammerblade(), target.hammerblade())
    assert output_h.device == torch.device("hammerblade")
    assert torch.allclose(output, output_h.cpu())
    output.backward()
    output_h.backward()
    assert input.grad is not None
    assert input_h.grad is not None
    assert torch.allclose(input.grad, input_h.grad)

def test_torch_nn_NLLLoss_sum_back_2():
    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss(reduction='sum')
    init = np.random.rand(3, 5)
    input = torch.tensor(init, dtype=torch.float, requires_grad=True)
    input_h = torch.tensor(init, dtype=torch.float, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    output = loss(m(input), target)
    output_h = loss(m(input_h).hammerblade(), target.hammerblade())
    assert output_h.device == torch.device("hammerblade")
    assert torch.allclose(output, output_h.cpu())
    output.backward()
    output_h.backward()
    assert input.grad is not None
    assert input_h.grad is not None
    assert torch.allclose(input.grad, input_h.grad)

def test_torch_nn_NLLLoss_none_back_1():
    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss(reduction='none')
    input = torch.tensor([[-3.5164, -1.9673, -4.7514, -0.2075, -4.6912],
                         [-2.9321, -1.5085, -0.8699, -1.4016, -2.8089],
                         [-2.6460, -1.9800, -1.4818, -1.8358, -0.9057]],
                         requires_grad=True)
    input_h = torch.tensor([[-3.5164, -1.9673, -4.7514, -0.2075, -4.6912],
                           [-2.9321, -1.5085, -0.8699, -1.4016, -2.8089],
                           [-2.6460, -1.9800, -1.4818, -1.8358, -0.9057]],
                           requires_grad=True)
    target = torch.tensor([1, 0, 4])
    grad = torch.tensor([1., 2., 3.])
    output = loss(m(input), target)
    output_h = loss(m(input_h).hammerblade(), target.hammerblade())
    assert output_h.device == torch.device("hammerblade")
    assert torch.allclose(output, output_h.cpu())
    output.backward(grad)
    output_h.backward(grad.hammerblade())
    assert input.grad is not None
    assert input_h.grad is not None
    assert torch.allclose(input.grad, input_h.grad)

def test_torch_nn_NLLLoss_none_back_2():
    m = nn.LogSoftmax(dim=1)
    loss = nn.NLLLoss(reduction='none')
    init = np.random.rand(3, 5)
    input = torch.tensor(init, dtype=torch.float, requires_grad=True)
    input_h = torch.tensor(init, dtype=torch.float, requires_grad=True)
    target = torch.tensor([1, 0, 4])
    grad = torch.tensor([1., 2., 3.])
    output = loss(m(input), target)
    output_h = loss(m(input_h).hammerblade(), target.hammerblade())
    assert output_h.device == torch.device("hammerblade")
    assert torch.allclose(output, output_h.cpu())
    output.backward(grad)
    output_h.backward(grad.hammerblade())
    assert input.grad is not None
    assert input_h.grad is not None
    assert torch.allclose(input.grad, input_h.grad)
