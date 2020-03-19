"""
Tests on torch.nn.Dropout
03/16/2020 Lin Cheng (lc873@cornell.edu)
"""
import torch
import torch.nn as nn

def test_torch_nn_dropout_1():
    dropout = nn.Dropout(0.5)
    x = torch.ones(10).hammerblade()
    x_d = dropout(x)
    assert x_d.device == torch.device("hammerblade")
    _sum = 0
    for i in x_d.cpu():
        assert (i == 0 or i == 2)
        _sum += i
    assert _sum != 0
    assert _sum != 20

def test_torch_nn_dropout_2():
    dropout = nn.Dropout(0.5)
    x = torch.randn(10).hammerblade()
    x_d = dropout(x)
    assert x_d.device == torch.device("hammerblade")
    has_zero = False
    for i in x_d.cpu():
        if i == 0:
            has_zero = True
    assert has_zero
    assert not torch.allclose(x_d.cpu(), x.cpu())

def test_torch_nn_dropout_3():
    dropout = nn.Dropout(0.5)
    x = torch.randn(10).hammerblade()
    x_d = dropout(x)
    assert x_d.device == torch.device("hammerblade")
    x_d = x_d.cpu()
    x = x.cpu()
    i = 0
    while(i < x.numel()):
        assert (torch.allclose(x_d[i], x[i] * 2) or x_d[i] == 0)
        i += 1
    assert not torch.allclose(x_d, x)

def test_torch_nn_dropout_4():
    dropout = nn.Dropout(0.25)
    x = torch.randn(10).hammerblade()
    x_d = dropout(x)
    assert x_d.device == torch.device("hammerblade")
    x_d = x_d.cpu()
    x = x.cpu()
    i = 0
    while(i < x.numel()):
        assert (torch.allclose(x_d[i], x[i] * (1 / 0.75)) or x_d[i] == 0)
        i += 1
    assert not torch.allclose(x_d, x)
