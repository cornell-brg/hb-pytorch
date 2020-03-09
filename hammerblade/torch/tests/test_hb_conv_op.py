import torch
import torch.nn.functional as F

def test_conv2d_1():
    kernel = torch.rand(1, 1, 3, 3)
    inputs = torch.rand(1, 1, 5, 5)

    kernel_hb = kernel.hammerblade()
    inputs_hb = inputs.hammerblade()

    conv_result_hb = F.conv2d(inputs_hb, kernel_hb, padding=1)
    conv_result = F.conv2d(inputs, kernel, padding=1)

    assert torch.allclose(conv_result, conv_result_hb.cpu())

def test_conv2d_2():
    kernel = torch.rand(1, 2, 3, 3)
    inputs = torch.rand(1, 2, 5, 5)

    kernel_hb = kernel.hammerblade()
    inputs_hb = inputs.hammerblade()

    conv_result_hb = F.conv2d(inputs_hb, kernel_hb, padding=1)
    conv_result = F.conv2d(inputs, kernel, padding=1)

    assert torch.allclose(conv_result, conv_result_hb.cpu())

def test_conv2d_3():
    kernel = torch.rand(2, 1, 3, 3)
    inputs = torch.rand(1, 1, 5, 5)

    kernel_hb = kernel.hammerblade()
    inputs_hb = inputs.hammerblade()

    conv_result_hb = F.conv2d(inputs_hb, kernel_hb, padding=1)
    conv_result = F.conv2d(inputs, kernel, padding=1)

    assert torch.allclose(conv_result, conv_result_hb.cpu())

def test_conv2d_4():
    kernel = torch.rand(5, 3, 3, 3)
    inputs = torch.rand(2, 3, 5, 5)

    kernel_hb = kernel.hammerblade()
    inputs_hb = inputs.hammerblade()

    conv_result_hb = F.conv2d(inputs_hb, kernel_hb, padding=(1, 2), stride=(1, 2))
    conv_result = F.conv2d(inputs, kernel, padding=(1, 2), stride=(1, 2))

    assert torch.allclose(conv_result, conv_result_hb.cpu())
