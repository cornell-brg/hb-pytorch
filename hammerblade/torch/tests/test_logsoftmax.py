"""
Unit tests for log_softmax operator
03/20/2020 Bandhav Veluri
"""

import torch
import torch.nn.functional as F

def _test_log_softmax(x, dim):
    x_hb = x.hammerblade()

    y = F.log_softmax(x, dim)
    y_hb = F.log_softmax(x_hb, dim)

    assert torch.allclose(y, y_hb.cpu())

def test_log_softmax_1():
    x = torch.rand(2, 3)
    dim = 1
    _test_log_softmax(x, dim)
