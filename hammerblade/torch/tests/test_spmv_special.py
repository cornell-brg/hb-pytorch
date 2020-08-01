import torch 
import torch.nn.functional as F

def test_spmv_1():
    m = torch.nn.Threshold(0.95, 0)
    input = torch.rand(10, 20)
    xv = torch.rand(20)
    xs = m(input).to_sparse()
    print(xs)
    
    xr = torch.mv(xs, xv)
    print(xr)
    hb_xv = xv.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mv(hb_xs, hb_xv)
    print(hb_xr)
    cpu_r = hb_xr.to("cpu")
    
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_spmv_2():
    m = torch.nn.Threshold(0.9, 0)
    input = torch.rand(10, 100)
    xv = torch.rand(100)
    xs = m(input).to_sparse()
    print(xs)

    xr = torch.mv(xs, xv)
    print(xr)
    hb_xv = xv.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mv(hb_xs, hb_xv)
    print(hb_xr)
    cpu_r = hb_xr.to("cpu")

    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

