"""
Unit tests for log_softmax operator
03/20/2020 Bandhav Veluri
"""

import torch
import torch.nn.functional as F

def test_log_softmax_1():
    x = torch.rand(2, 3)
    x_hb = x.hammerblade()

    y = F.log_softmax(x, dim=1)
    print(y)

    y_hb = F.log_softmax(x_hb, dim=1)
    print(y_hb)

    assert(y, y_hb.cpu())
