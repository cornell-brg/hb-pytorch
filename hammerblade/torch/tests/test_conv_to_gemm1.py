import torch
import torch.nn as nn

def convert_input_dense(x):
    stride = (1, 1)
    kernel_size = (5, 5)
    assert x.dim() == 4
    torch.aten_profiler_start()
    x = x.unfold(2, kernel_size[0], stride[0])
    x = x.unfold(3, kernel_size[1], stride[1])
    torch.aten_profiler_end()
    x = torch.flatten(x, start_dim = 4)
    x = x.transpose(1, 3).transpose(1, 2)
    x = torch.flatten(x, start_dim = 3)
    x = torch.flatten(x, start_dim = 0, end_dim = 2).t().contiguous()
    return x

def test_lenet5_cpu_to_cpu_conv1():
    in_channel = 1
    out_channel = 20
    
    x = torch.rand(1, in_channel, 32, 32)
    conv = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 5, stride = 1, bias = False)
    out1 = conv(x)

    x = convert_input_dense(x)
    y = conv.weight.data.view(out_channel, -1)
    out2 = torch.mm(y, x)

    out2 = out2.view(1, out_channel, 28, 28)
    assert torch.allclose(out1, out2) 

def test_lenet5_cpu_to_cpu_conv2():
    in_channel = 6
    out_channel = 16
    x = torch.rand(1, in_channel, 14, 14)
    conv = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 5, stride = 1, bias=False)
    out1 = conv(x)

    x = convert_input_dense(x)
    y = conv.weight.data.view(out_channel, -1)
    out2 = torch.mm(y, x)
    out2 = out2.view(1, out_channel, 10, 10)

def test_lenet5_cpu_to_hb_conv1():
    in_channel = 1
    out_channel = 20
    x = torch.rand(1, in_channel, 32, 32)
    conv = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 5, stride = 1, bias=False)
    cpu_out1 = conv(x)

    hb_x = x.hammerblade()
    hb_x = convert_input_dense(x)
    hb_y = conv.weight.data.view(out_channel, -1).hammerblade()
    hb_out = torch.mm(hb_y, hb_x)
    hb_out = hb_out.view(1, out_channel, 28, 28)
    cpu_out2 = hb_out.cpu()
    assert torch.allclose(out1, cpu_out2)

def test_lenet5_cpu_to_hb_conv2():
    in_channel = 6
    out_channel = 16
    x = torch.rand(1, in_channel, 14, 14)
    conv = nn.Conv2d(in_channels = in_channel, out_channels = out_channel, kernel_size = 5, stride = 1, bias=False)
    cpu_out1 = conv(x)

    hb_x = x.hammerblade()
    hb_x = convert_input_dense(x)
    hb_y = conv.weight.data.view(out_channel, -1).hammerblade()
    hb_out = torch.mm(hb_y, hb_x)
    hb_out = hb_out.view(1, out_channel, 10, 10)
    cpu_out2 = hb_out.cpu()
    assert torch.allclose(out1, cpu_out2)

     
