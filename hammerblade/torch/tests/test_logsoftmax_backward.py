"""
Tests on torch.nn.LogSoftMax backward
03/26/2020 Lin Cheng (lc873@cornell.edu)
"""

import torch
import torch.nn.functional as F
import numpy as np

torch.manual_seed(42)
np.random.seed(42)

def _test_log_softmax_back(init, dim):
    x = torch.tensor(init, dtype=torch.float, requires_grad=True)
    x_hb = torch.tensor(init, dtype=torch.float, requires_grad=True)

    y = F.log_softmax(x, dim)
    y_hb = F.log_softmax(x_hb.hammerblade(), dim)

    assert y_hb.device == torch.device("hammerblade")
    assert torch.allclose(y, y_hb.cpu(), atol=1e-7)

    y.backward(y * -1.0)
    y_hb.backward(y_hb * -1.0)

    assert x.grad is not None
    assert x_hb.grad is not None
    assert torch.allclose(x.grad, x_hb.grad, atol=1e-6)

def test_log_softmax_back_1():
    init = np.random.rand(2, 3)
    dim = 1
    _test_log_softmax_back(init, dim)

def test_log_softmax_back_2():
    init = np.random.rand(2, 3)
    dim = 0
    _test_log_softmax_back(init, dim)

def test_log_softmax_back_3():
    init = np.random.rand(5)
    dim = 0
    _test_log_softmax_back(init, dim)

def test_log_softmax_back_4():
    init = np.random.rand(1, 6)
    dim = 1
    _test_log_softmax_back(init, dim)

def test_log_softmax_back_5():
    init = np.random.rand(1, 6)
    dim = 0
    _test_log_softmax_back(init, dim)

def test_log_softmax_back_6():
    init = np.random.rand(2, 3, 3, 5)

    for dim in range(4):
        _test_log_softmax_back(init, dim)
