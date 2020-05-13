"""
Tests on torch.eye (identity kernel)
04/10/2020 Michelle Chao (mc2244@cornell.edu)
"""
import torch

def _test_torch_eye_square(n):
    assert torch.allclose(torch.eye(n), torch.eye(n, device=torch.device("hammerblade")).cpu())

def _test_torch_eye(n, m):
    assert torch.allclose(torch.eye(n, m), torch.eye(n, m, device=torch.device("hammerblade")).cpu())

def test_torch_eye_1():
    _test_torch_eye_square(4)

def test_torch_eye_2():
    _test_torch_eye_square(1)

def test_torch_eye_3():
    _test_torch_eye_square(16)

def test_torch_eye_4():
    _test_torch_eye(3, 4)

def test_torch_eye_5():
    _test_torch_eye(4, 1)

def test_torch_eye_6():
    _test_torch_eye(3, 3)
