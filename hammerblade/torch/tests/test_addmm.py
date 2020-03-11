"""
Unit tests for torch.addmm kernel
03/09/2020 Lin Cheng (lc873@cornell.edu)
"""
import torch
import pytest

@pytest.mark.skip(reason="not yet implemented")
def test_torch_addmm_1():
    M = torch.ones(2, 3)
    M_h = M.hammerblade()
    mat1 = torch.ones(2, 3)
    mat1_h = mat1.hammerblade()
    mat2 = torch.ones(3, 3)
    mat2_h = mat2.hammerblade()
    out = torch.addmm(M, mat1, mat2)
    out_h = torch.addmm(M_h, mat1_h, mat2_h)
    assert out_h.device == torch.device("hammerblade")
    assert torch.equal(out_h.cpu(), out)

@pytest.mark.skip(reason="not yet implemented")
def test_torch_addmm_2():
    M = torch.randn(2, 3)
    M_h = M.hammerblade()
    mat1 = torch.randn(2, 3)
    mat1_h = mat1.hammerblade()
    mat2 = torch.randn(3, 3)
    mat2_h = mat2.hammerblade()
    out = torch.addmm(M, mat1, mat2)
    out_h = torch.addmm(M_h, mat1_h, mat2_h)
    assert out_h.device == torch.device("hammerblade")
    assert torch.equal(out_h.cpu(), out)
