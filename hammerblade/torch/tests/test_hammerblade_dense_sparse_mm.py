import torch 

def test_customized_sparse_dense_tensor_mm():

    i = torch.LongTensor([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 2, 2, 3, 1, 3]])
    v = torch.ones(8)
    x = torch.sparse.FloatTensor(i, v, torch.Size([4, 4]))
    xs = x.coalesce()
    xd = torch.ones(4, 1)
    xr = torch.mm(xs, xd)
   
    hb_xs = xs.hammerblade()
    hb_xd = xd.hammerblade()
    hb_xr = torch.mm(hb_xs, hb_xd)
    print(hb_xr)
    cpu_r = hb_xr.cpu()
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_random_sparse_dense_mm1():
    xd = torch.rand(100, 1)
    xs = torch.rand(1, 100).to_sparse()
    xr = torch.mm(xs, xd)

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mm(hb_xs, hb_xd)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_random_sparse_dense_mm2():
    xd = torch.rand(16, 1)
    xs = torch.rand(16, 16).to_sparse()
    xr = torch.mm(xs, xd)
    print(xr)
    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mm(hb_xs, hb_xd)
    print(hb_xr)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_random_sparse_dense_mm3():
    xd = torch.rand(16, 5)
    xs = torch.rand(4, 16).to_sparse()
    xr = torch.mm(xs, xd)
    print(xr)

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mm(hb_xs, hb_xd)
    print(hb_xr)
    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr) 

def test_random_sparse_dense_mm4():
    xd = torch.rand(1, 500)
    xs = torch.rand(100, 500).to_sparse()
    xr = torch.mm(xs, xd.t())
    print(xr)

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mm(hb_xs, hb_xd.t())
    print(hb_xr)
    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)
