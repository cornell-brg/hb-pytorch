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

def _test_batch_norm(inputs):
    inputs_hb = hbutils.init_hb_tensor(inputs)

    out = F.batch_norm(inputs, None, None, training=True)
    out_hb = F.batch_norm(inputs_hb, None, None, training=True)

    assert torch.allclose(out, out_hb, atol=1e-5)

def test_batch_norm2d_1():
    inputs = torch.rand(3, 1, 3, 3, requires_grad=True)

    _test_batch_norm(inputs)
