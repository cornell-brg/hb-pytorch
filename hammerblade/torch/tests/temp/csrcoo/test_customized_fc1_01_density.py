import torch

def test_customize_fc_01_density():
    m = torch.nn.Threshold(0.9, 0)
    input = torch.rand(40, 250)
    xd = torch.rand(1, 250)
    xs = m(input).to_sparse()
    print(xs._nnz())
    xr = torch.mm(xs, xd.t())
    print(xr)
    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mm(hb_xs, hb_xd.t())
    print(hb_xr)
    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)
