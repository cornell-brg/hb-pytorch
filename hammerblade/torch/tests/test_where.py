"""
Tests on where operation
09/20/2021 Niklas Schmelzle (jms854@cornell.edu)
"""
import torch
import torch.nn as nn

def _test_where(condition_ten, x_ten, y_ten):
    xh_ten = x_ten.hammerblade()
    yh_ten = y_ten.hammerblade()
    conditionh_ten = condition_ten.hammerblade()

    out = torch.where(condition_ten, x_ten, y_ten)
    outh = torch.where(conditionh_ten, xh_ten, yh_ten)

    assert outh.device == torch.device("hammerblade")
    assert torch.allclose(out, outh.cpu())


# condition is bool
def test_where1():
    x = torch.tensor([[-0.5,  2.0,  0.1],
                      [ 1.1, -1.1, -7.0],
                      [ 3.2, -2.1, 42.0]])
    y = torch.ones(x.shape)
    condition_bool = torch.tensor([[True, False, True],
                              [False, True, False],
                              [True, False, True]], dtype=torch.bool)
    _test_where(condition_bool, x, y)

def test_where2():
    x = torch.tensor([[-0.5,  2.0,  0.1],
                      [ 1.1, -1.1, -7.0],
                      [ 3.2, -2.1, 42.0]])
    y = torch.ones(x.shape)
    condition_bool = x > 0.1
    _test_where(condition_bool, x, y)

def test_where3():
    x = torch.rand(50, 50)
    y = torch.rand(x.shape)
    condition_bool = x > 0.5

    _test_where(condition_bool, x, y)


# condition is uint8
def test_where4():
    x = torch.tensor([[-0.5,  2.0,  0.1],
                      [ 1.1, -1.1, -7.0],
                      [ 3.2, -2.1, 42.0]])
    y = torch.ones(x.shape)
    condition_uint8 = torch.tensor([[0, 8, 0],
                                    [7, 0, 9],
                                    [0, 8, 0]], dtype=torch.uint8)
    _test_where(condition_uint8, x, y)

def test_where5():
    x = torch.rand(50, 50)
    y = torch.rand(x.shape)
    condition_uint8 = torch.randint(0, 10, (x.shape), dtype=torch.uint8)

    _test_where(condition_uint8, x, y)
