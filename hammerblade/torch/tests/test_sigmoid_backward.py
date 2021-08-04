"""
Tests on sigmoid_backward
08/04/2021 Niklas Schmelzle (jms854@cornell.edu)
"""
import torch
import torch.nn as nn

def test_sigmoid_backward_scalar():
    sigmoid = nn.Sigmoid()

    inp = torch.tensor([0.0], requires_grad=True)
    inp_h = torch.tensor([0.0], device="hammerblade", requires_grad=True)

    out = sigmoid(inp)
    out_h = sigmoid(inp_h)

    out.backward()
    out_h.backward()

    assert out_h.device == torch.device("hammerblade")
    assert torch.allclose(inp.grad, inp_h.grad.cpu())

def test_sigmoid_backward_vector():
    sigmoid = nn.Sigmoid()

    inp = torch.tensor([0.5, 0.6, 0.7], requires_grad=True)
    inp_h = torch.tensor([0.5, 0.6, 0.7], device="hammerblade", requires_grad=True)

    out = sigmoid(inp)
    out_h = sigmoid(inp_h)

    grad_output = torch.tensor([0.7, 0.8, 0.9])
    grad_output_h = grad_output.hammerblade()
    out.backward(grad_output)
    out_h.backward(grad_output_h)

    assert out_h.device == torch.device("hammerblade")
    assert torch.allclose(inp.grad, inp_h.grad.cpu())

# TODO: Hypothesis?
