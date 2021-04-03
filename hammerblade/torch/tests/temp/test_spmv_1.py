"""
Unit tests for torch.mv kernel, specifically, the spmv operation
07/18/2020 Zhongyuan Zhao (zz546@cornell.edu)
"""
import torch 
import torch.nn.functional as F

m01 = torch.nn.Threshold(0.9, 0)

def test_customized_spmv():

    i = torch.LongTensor([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 2, 2, 3, 1, 3]])
    v = torch.ones(8)
    x = torch.sparse.FloatTensor(i, v, torch.Size([4, 4]))
    xs = x.coalesce()
    xv = torch.ones(4)
    xr = torch.mv(xs, xv)
   
    hb_xs = xs.hammerblade()
    hb_xv = xv.hammerblade()
    hb_xr = torch.mv(hb_xs, hb_xv)
    cpu_r = hb_xr.cpu()
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

