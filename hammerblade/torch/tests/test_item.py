"""
Unit tests for torch.tensor.item
03/29/2020 Lin Cheng (lc873@cornell.edu)
"""

import math
import torch
import pytest

def _test_torch_tensor_item(tensor):
    h = tensor.hammerblade()
    scalar = h.item()
    assert math.isclose(scalar, tensor[0], rel_tol=1e-08)

def test_torch_tensor_item_1():
    x = torch.ones(1)
    _test_torch_tensor_item(x)

def test_torch_tensor_item_2():
    x = torch.rand(1)
    _test_torch_tensor_item(x)

def test_torch_tensor_item_3():
    x = torch.randn(1)
    _test_torch_tensor_item(x)

@pytest.mark.xfail
def test_torch_tensor_item_F():
     x = torch.ones(10)
     _test_torch_tensor_item(x)
