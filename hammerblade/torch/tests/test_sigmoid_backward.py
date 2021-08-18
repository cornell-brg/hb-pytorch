"""
Tests on sigmoid_backward
08/04/2021 Niklas Schmelzle (jms854@cornell.edu)
"""
import torch
import torch.nn as nn

from random import randrange
import numpy as np

def _test_sigmoid_backward(inp, grad_out=[1.]):
    sigmoid = nn.Sigmoid()

    inp_ten = torch.tensor(inp, requires_grad=True)
    inp_h_ten = torch.tensor(inp, device="hammerblade", requires_grad=True)

    out_ten = sigmoid(inp_ten)
    out_h_ten = sigmoid(inp_h_ten)

    grad_output_ten = torch.tensor(grad_out)
    grad_output_h_ten = grad_output_ten.hammerblade()

    out_ten.backward(grad_output_ten)
    out_h_ten.backward(grad_output_h_ten)

    assert out_h_ten.device == torch.device("hammerblade")
    assert torch.allclose(inp_ten.grad, inp_h_ten.grad.cpu())


def test_sigmoid_backward_scalar():
    _test_sigmoid_backward([0.0])

def test_sigmoid_backward_vector():
    _test_sigmoid_backward([0.5, 0.6, 0.7], [0.7, 0.8, 0.9])

def test_sigmoid_backward_random():
    x_size = randrange(1, 50)
    y_size = randrange(1, 10)

    inp = [1.0]
    grad_out = [1.0]
    if y_size == 1:
        inp = np.random.rand(x_size).tolist()
        grad_out = np.random.rand(x_size).tolist()
    else:
        inp = np.random.rand(y_size, x_size).tolist()
        grad_out = np.random.rand(y_size, x_size).tolist()

    _test_sigmoid_backward(inp, grad_out)

# TODO: Hypothesis?
