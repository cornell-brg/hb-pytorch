import torch 

#We select three test functions to test torch.mm and torch.sparse.mm for sparse mm tensor operation

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
    
    cpu_r = hb_xr.cpu()
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_customized_sparse_dense_sparse_mm():

    i = torch.LongTensor([[0, 0, 1, 1, 2, 2, 3, 3], [0, 1, 1, 2, 2, 3, 1, 3]])
    v = torch.ones(8)
    x = torch.sparse.FloatTensor(i, v, torch.Size([4, 4]))
    xs = x.coalesce()
    xd = torch.ones(4, 1)
    xr = torch.sparse.mm(xs, xd)

    hb_xs = xs.hammerblade()
    hb_xd = xd.hammerblade()
    hb_xr = torch.sparse.mm(hb_xs, hb_xd)

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

def test_random_sparse_dense_sparse_mm1():
    xd = torch.rand(100, 1)
    xs = torch.rand(1, 100).to_sparse()
    xr = torch.sparse.mm(xs, xd)

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.sparse.mm(hb_xs, hb_xd)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_random_sparse_dense_mm2_hybrid1():
    xd = torch.rand(16, 1)
    xs = torch.rand(16, 16).to_sparse()
    xr = torch.sparse.mm(xs, xd)

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mm(hb_xs, hb_xd)

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_random_sparse_dense_mm3_hybrid2():
    xd = torch.rand(16, 5)
    xs = torch.rand(4, 16).to_sparse()
    xr = torch.mm(xs, xd)
    print(xr)

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.sparse.mm(hb_xs, hb_xd)
    print(hb_xr)
    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr) 

def test_lenet5_fc1_01_density_mm():
    m = torch.nn.Threshold(0.9, 0)
    input = torch.rand(84, 120)
    xd = torch.rand(10, 120)
    xs = m(input).to_sparse()
    print(xs._nnz())
    xr = torch.mm(xs, xd.t())

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mm(hb_xs, hb_xd.t())

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_lenet5_fc1_02_density_sparse_mm():
    m = torch.nn.Threshold(0.8, 0)
    input = torch.rand(84, 120)
    xd = torch.rand(10, 120)
    xs = m(input).to_sparse()
    print(xs._nnz())
    xr = torch.sparse.mm(xs, xd.t())

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.sparse.mm(hb_xs, hb_xd.t())

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_lenet5_fc1_03_density_hybrid1():
    m = torch.nn.Threshold(0.7, 0)
    input = torch.rand(84, 120)
    xd = torch.rand(10, 120)
    xs = m(input).to_sparse()
    print(xs._nnz())
    xr = torch.sparse.mm(xs, xd.t())

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mm(hb_xs, hb_xd.t())

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_lenet5_fc1_04_density_hybrid2():
    m = torch.nn.Threshold(0.6, 0)
    input = torch.rand(84, 120)
    xd = torch.rand(10, 120)
    xs = m(input).to_sparse()
    print(xs._nnz())
    xr = torch.mm(xs, xd.t())

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.sparse.mm(hb_xs, hb_xd.t())

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_lenet5_fc1_05_density_mm():
    m = torch.nn.Threshold(0.5, 0)
    input = torch.rand(84, 120)
    xd = torch.rand(10, 120)
    xs = m(input).to_sparse()
    print(xs._nnz())
    xr = torch.mm(xs, xd.t())

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mm(hb_xs, hb_xd.t())

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_lenet5_fc2_01_density_mm():
    m = torch.nn.Threshold(0.9, 0)
    input = torch.rand(10, 500)
    xd = torch.rand(2, 500)
    xs = m(input).to_sparse()
    print(xs._nnz())
    xr = torch.mm(xs, xd.t())

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mm(hb_xs, hb_xd.t())

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_lenet5_fc2_02_density_sparse_mm():
    m = torch.nn.Threshold(0.8, 0)
    input = torch.rand(10, 500)
    xd = torch.rand(2, 500)
    xs = m(input).to_sparse()
    print(xs._nnz())
    xr = torch.sparse.mm(xs, xd.t())

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.sparse.mm(hb_xs, hb_xd.t())

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_lenet5_fc2_03_density_hybrid1():
    m = torch.nn.Threshold(0.7, 0)
    input = torch.rand(10, 500)
    xd = torch.rand(2, 500)
    xs = m(input).to_sparse()
    print(xs._nnz())
    xr = torch.sparse.mm(xs, xd.t())

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mm(hb_xs, hb_xd.t())

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_lenet5_fc2_04_density_hybrid2():
    m = torch.nn.Threshold(0.6, 0)
    input = torch.rand(10, 500)
    xd = torch.rand(2, 500)
    xs = m(input).to_sparse()
    print(xs._nnz())
    xr = torch.mm(xs, xd.t())

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.sparse.mm(hb_xs, hb_xd.t())

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)

def test_lenet5_fc2_05_density_mm():
    m = torch.nn.Threshold(0.5, 0)
    input = torch.rand(10, 500)
    xd = torch.rand(2, 500)
    xs = m(input).to_sparse()
    print(xs._nnz())
    xr = torch.mm(xs, xd.t())

    hb_xd = xd.hammerblade()
    hb_xs = xs.hammerblade()
    hb_xr = torch.mm(hb_xs, hb_xd.t())

    cpu_r = hb_xr.to("cpu")
    assert hb_xr.device == torch.device("hammerblade")
    assert torch.allclose(cpu_r, xr)
