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

def test_random_spmv1():
    xv = torch.rand(100)
    xs = torch.rand(1, 100)
    xs = m01(xs).to_sparse()

    xr = torch.mv(xs, xv)

    hb_xv = xv.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mv(hb_xs, hb_xv)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_lenet5_fc1_01_density_mv():

    xs = torch.rand(84, 120)
    xv = torch.rand(120)
    xs = m01(xs).to_sparse()
    xr = torch.mv(xs, xv)

    hb_xv = xv.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mv(hb_xs, hb_xv)
    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_lenet5_fc1_02_density_mv():
    m = torch.nn.Threshold(0.8, 0)
    input = torch.rand(84, 120)
    xv = torch.rand(120)
    xs = m(input).to_sparse()
   
    xr = torch.mv(xs, xv)

    hb_xv = xv.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mv(hb_xs, hb_xv)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_lenet5_fc1_04_density_mv():
    m = torch.nn.Threshold(0.6, 0)
    input = torch.rand(84, 120)
    xv = torch.rand(120)
    xs = m(input).to_sparse()

    xr = torch.mv(xs, xv)

    hb_xv = xv.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mv(hb_xs, hb_xv)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_lenet5_fc1_05_density_mv():
    m = torch.nn.Threshold(0.5, 0)
    input = torch.rand(84, 120)
    xv = torch.rand(120)
    xs = m(input).to_sparse()
    xr = torch.mv(xs, xv)

    hb_xv = xv.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mv(hb_xs, hb_xv)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_lenet5_fc2_01_density_mv():
    m = torch.nn.Threshold(0.9, 0)
    input = torch.rand(10, 500)
    xv = torch.rand(500)
    xs = m(input).to_sparse()

    xr = torch.mv(xs, xv) 
   
    hb_xv = xv.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mv(hb_xs, hb_xv)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_lenet5_fc2_02_density_sparse_mv():
    m = torch.nn.Threshold(0.8, 0)
    input = torch.rand(10, 500)
    xv = torch.rand(500)
    xs = m(input).to_sparse()

    xr = torch.mv(xs, xv)

    hb_xv = xv.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mv(hb_xs, hb_xv)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_lenet5_fc2_03_density_mv():
    m = torch.nn.Threshold(0.7, 0)
    input = torch.rand(10, 500)
    xv = torch.rand(500)
    xs = m(input).to_sparse()

    xr = torch.mv(xs, xv)

    hb_xv = xv.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mv(hb_xs, hb_xv)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_lenet5_fc2_04_density_mv():
    m = torch.nn.Threshold(0.6, 0)
    input = torch.rand(10, 500)
    xv = torch.rand(500)
    xs = m(input).to_sparse()

    xr = torch.mv(xs, xv)

    hb_xv = xv.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mv(hb_xs, hb_xv)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_lenet5_fc2_05_density_mv():
    m = torch.nn.Threshold(0.5, 0)
    input = torch.rand(10, 500)
    xv = torch.rand(500)
    xs = m(input).to_sparse()

    xr = torch.mv(xs, xv)

    hb_xv = xv.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mv(hb_xs, hb_xv)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)
