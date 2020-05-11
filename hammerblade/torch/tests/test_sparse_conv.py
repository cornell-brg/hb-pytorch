import torch
import torch.nn.functional as F

m01 = torch.nn.Threshold(0.9, 0)
m02 = torch.nn.Threshold(0.8, 0)
m03 = torch.nn.Threshold(0.7, 0)

def convert_dense_input(x):
    stride = (1, 1)
    kernel_size = (5, 5)
    assert x.dim() == 4
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    x = torch.flatten(x, start_dim = 4)
    x = x.transpose(1, 3).transpose(1, 2)
    x = torch.flatten(x, start_dim = 3)
    x = torch.flatten(x, start_dim = 0, end_dim = 2).t().contiguous()
    return x

def test_lenet5_sparse01_conv1():
    
    di = torch.rand(1, 1, 32, 32)
    dw = torch.rand(6, 1, 5, 5)
    dw = m01(dw)
    sw = dw.to_sparse()

    cpu_i = convert_dense_input(di)
    cpu_sw = dw.view(6, -1).to_sparse()
    cpu_out = torch.mm(cpu_sw, cpu_i)
    out1 = cpu_out.view(1, 6, 28, 28)
    print(out1)

    hb_i = di.hammerblade()
    hb_sw = dw.hammerblade()
    hb_out = F.conv2d(hb_i, hb_sw, bias = None, stride = 1, padding = 0, dilation = 1)
    print(hb_out)
    out2 = hb_out.cpu()
    
    assert torch.allclose(out1, out2) 
