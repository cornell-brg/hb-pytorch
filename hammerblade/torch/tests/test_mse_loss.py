"""
Tests on mse_loss
08/10/2021 Niklas Schmelzle (jms854@cornell.edu)
"""
import torch
import torch.nn as nn

from random import randrange
import numpy as np

def _test_mse_loss(inp_list, target_list, target_requires_grad=False):
    loss = nn.MSELoss()

    inp = torch.tensor(inp_list)
    inph = inp.hammerblade()
    target = torch.tensor(target_list, requires_grad=target_requires_grad)
    targeth = target.hammerblade()

    output = loss(inp, target)
    outputh = loss(inph, targeth)
    print()
    print("output: ")
    print(output)
    print("outputh: ")
    print(outputh)
    assert outputh.device == torch.device("hammerblade")
    assert torch.allclose(output, outputh.cpu())


# target does not require gradient
def test_mse_loss1():
    _test_mse_loss([2.0], [2.5], False)

def test_mse_loss2():
    _test_mse_loss([1.0, 7.0, -3.0], [2.0, 5.0, -1.0], False)

def test_mse_loss3():
    x_size = randrange(50)
    y_size = randrange(10)

    inp = [1.0]
    target = [1.0]
    if y_size == 1:
        inp = np.random.rand(x_size).tolist()
        target = np.random.rand(x_size).tolist()
    else:
        inp = np.random.rand(y_size, x_size).tolist()
        target = np.random.rand(y_size, x_size).tolist()

    _test_mse_loss(inp, target, False)


# target requires gradient
# pow (tensor-scalar) kernel is also utilized
def test_mse_loss_grad1():
    _test_mse_loss([2.0], [2.5], True)

def test_mse_loss_grad2():
    _test_mse_loss([1.0, 7.0, -3.0], [2.0, 5.0, -1.0], True)

def test_mse_loss_grad3():
    x_size = randrange(50)
    y_size = randrange(10)

    inp = [1.0]
    target = [1.0]
    if y_size == 1:
        inp = np.random.rand(x_size).tolist()
        target = np.random.rand(x_size).tolist()
    else:
        inp = np.random.rand(y_size, x_size).tolist()
        target = np.random.rand(y_size, x_size).tolist()

    _test_mse_loss(inp, target, True)
