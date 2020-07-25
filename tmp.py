import torch
import torch.nn.functional as F
import torch.autograd.profiler as torchprof
import torch.hammerblade.profiler as hbprof

def test_conv2d_3():
    inputs = torch.rand(8, 16, 32, 32)
    kernel = torch.rand(32, 16, 3, 3)
    kernel_hb = kernel.hammerblade()
    inputs_hb = inputs.hammerblade()

    conv_result = F.conv2d(inputs, kernel)
    conv_result_hb = F.conv2d(inputs_hb, kernel_hb)
    print(conv_result.shape)
    assert torch.allclose(conv_result, conv_result_hb.cpu())

def test_vec_add():
    """
    Single batch, single channel
    """
    a = torch.rand(100000)
    b = torch.rand(100000)
    a_hb = a.hammerblade()
    b_hb = b.hammerblade()

    c = torch.add(a, b)
    c_hb = torch.add(a_hb, b_hb)
    assert torch.allclose(c, c_hb.cpu())

def test_copy():
    print(torch.rand(2,3).hammerblade())

test_conv2d_3()
