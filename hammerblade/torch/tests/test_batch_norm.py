"""
Unit tests for batch_norm operator
07/11/2020 Bandhav Veluri
"""
import torch
import torch.nn.functional as F
import random
import hbutils

torch.manual_seed(42)
random.seed(42)

def _test_batch_norm(inputs, running_mean, running_var,
                     weights=None, bias=None, training=False):
    inputs_hb = hbutils.init_hb_tensor(inputs)
    running_mean_hb = hbutils.init_hb_tensor(running_mean)
    running_var_hb = hbutils.init_hb_tensor(running_var)
    weights_hb = hbutils.init_hb_tensor(weights)
    bias_hb = hbutils.init_hb_tensor(bias)

    out = F.batch_norm(inputs, running_mean, running_var,
                       weights, bias, training)
    out_hb = F.batch_norm(inputs_hb, running_mean_hb, running_var_hb,
                          weights_hb, bias_hb, training)

    assert torch.allclose(out, out_hb.cpu(), atol=1e-4)
    assert torch.allclose(running_mean, running_mean_hb.cpu(), atol=1e-4)
    assert torch.allclose(running_var, running_var_hb.cpu(), atol=1e-4)

def test_batch_norm2d_eval_1():
    inputs = torch.rand(1, 3, 3, 3, requires_grad=True)
    running_mean = torch.rand(3)
    running_var = torch.rand(3)

    _test_batch_norm(inputs, running_mean, running_var)

def test_batch_norm2d_train_1():
    inputs = torch.rand(1, 3, 3, 3, requires_grad=True)
    running_mean = torch.rand(3)
    running_var = torch.rand(3)

    _test_batch_norm(inputs, running_mean, running_var, training=True)

def test_batch_norm2d_eval_2():
    inputs = torch.rand(3, 3, 2, 2, requires_grad=True)
    running_mean = torch.rand(3)
    running_var = torch.rand(3)

    _test_batch_norm(inputs, running_mean, running_var)

def test_batch_norm2d_train_2():
    inputs = torch.rand(3, 3, 2, 2, requires_grad=True)
    running_mean = torch.rand(3)
    running_var = torch.rand(3)

    _test_batch_norm(inputs, running_mean, running_var, training=True)

def _test_BatchNorm2d(n, inputs):
    bn = torch.nn.BatchNorm2d(n)
    bn_hb = torch.nn.BatchNorm2d(n).hammerblade()

    out = bn(inputs)
    out_hb = bn_hb(inputs.hammerblade())

    torch.allclose(out, out_hb.cpu(), atol=1e-5)

def test_BatchNorm2d_1():
    _test_BatchNorm2d(4, torch.ones(2, 4, 3, 3))

def test_BatchNorm2d_2():
    _test_BatchNorm2d(4, torch.rand(2, 4, 3, 3))

def test_BatchNorm2d_3():
    _test_BatchNorm2d(4, torch.zeros(2, 4, 3, 3))
