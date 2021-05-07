"""
Unit tests for torch.mv kernel, specifically, the spmv operation
07/18/2020 Zhongyuan Zhao (zz546@cornell.edu)
"""
import torch 
import torch.nn.functional as F

m01 = torch.nn.Threshold(0.8, 0)

def test_random_spmv1():
    d1 = torch.rand(10, 20)
    d2 = torch.rand(10, 20)
    s1 = m01(d1).to_sparse()
    s2 = m01(d2).to_sparse()

    r = s1 * s2

    hb_s1 = s1.hammerblade()
    hb_s2 = s2.hammerblade()
    hb_r = hb_s1 * hb_s2

    cpu_r = hb_r.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

