"""
Unit tests for batch_norm operator
07/11/2020 Bandhav Veluri
"""
import torch
import torch.nn.functional as F
import random
import pytest
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
