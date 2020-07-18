import torch

def test_torch_mm_large():
    mat1 = torch.randn(16, 400)
    mat2 = torch.randn(400, 32)
    mat1_h = mat1.hammerblade()
    mat2_h = mat2.hammerblade()
    out = torch.mm(mat1, mat2)
    out_h = torch.mm(mat1_h, mat2_h)
    assert out_h.device == torch.device("hammerblade")
    assert torch.allclose(out_h.cpu(), out, rtol=1e-03, atol=1e-05)
