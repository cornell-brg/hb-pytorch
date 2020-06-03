"""
Tests on torch.mm
03/10/2020 Lin Cheng (lc873@cornell.edu)
"""
import torch
import random
import pytest

torch.manual_seed(42)
random.seed(42)

def test_torch_mm_transpose_1():
    mat1 = torch.randn(3, 4)
    mat2 = torch.randn(3, 5)
    mat1_h = mat1.hammerblade()
    mat2_h = mat2.hammerblade()
    out = torch.mm(mat1.t(), mat2)
    out_h = torch.mm(mat1_h.t(), mat2_h)
    assert out_h.device == torch.device("hammerblade")
    assert torch.allclose(out_h.cpu(), out)
