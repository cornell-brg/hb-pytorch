import torch
import torch.nn.functional as F

m01 = torch.nn.Threshold(0.9, 0)

def test_lenet5_sparse01_conv1_lower_mm():

    di = torch.rand(25, 784)
    dw = torch.rand(6, 25)
    dw = m01(dw)
    sw = dw.to_sparse()

    cpu_out = torch.sparse.mm(sw, di)

    hb_i = di.hammerblade()
    hb_sw = sw.hammerblade()
    hb_out = torch.sparse.mm(hb_sw, hb_i)
    out2 = hb_out.cpu()

    assert torch.allclose(cpu_out, out2)
