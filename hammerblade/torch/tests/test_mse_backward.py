"""
Tests on mse_backward operation
08/18/2021 Niklas Schmelzle (jms854@cornell.edu)
"""
import torch
import torch.nn as nn
from random import randrange
import numpy as np

def _test_mse_backward(inp_list, target_list, target_requires_grad=False):
    loss = nn.MSELoss()

    inp = torch.tensor(inp_list, requires_grad=True)
    inph = torch.tensor(inp_list, requires_grad=True, device='hammerblade')

    target = torch.tensor(target_list, requires_grad=target_requires_grad)
    targeth = torch.tensor(target_list, requires_grad=target_requires_grad, device='hammerblade')

    out = loss(inp, target)
    outh = loss(inph, targeth)

    print("out:")
    print(out)
    print(outh)

    out.backward()
    outh.backward()

    print("gradient:")
    print(inp.grad)
    print(inph.grad)

    assert outh.device == torch.device("hammerblade")
    assert torch.allclose(inp.grad, inph.grad.cpu())


def test_mse_backward1():
    inp_list = [3.7]
    target_list = [3.0]
    _test_mse_backward(inp_list, target_list)

def test_mse_backward2():
    inp_list = [3.7, 5.5, 27.0, 3.9, -2.2]
    target_list = [3.0, 6.7, 26.9, 5.0, -3.0]
    _test_mse_backward(inp_list, target_list)

def test_mse_backward3():
    list_size = randrange(1, 1000)
    inp_list = np.random.rand(list_size).tolist()
    target_list = np.random.rand(list_size).tolist()
    _test_mse_backward(inp_list, target_list)


# following tests do not require mse_backward kernel
# since forward pass is implemented as python pytorch kernel
def test_mse_backward_target_grad1():
    inp_list = [3.7]
    target_list = [3.0]
    _test_mse_backward(inp_list, target_list, target_requires_grad=True)

def test_mse_backward_target_grad2():
    inp_list = [3.7, 5.5, 27.0, 3.9, -2.2]
    target_list = [3.0, 6.7, 26.9, 5.0, -3.0]
    _test_mse_backward(inp_list, target_list, target_requires_grad=True)

def test_mse_backward_target_grad3():
    list_size = randrange(1, 1000)
    inp_list = np.random.rand(list_size).tolist()
    target_list = np.random.rand(list_size).tolist()
    _test_mse_backward(inp_list, target_list, target_requires_grad=True)
