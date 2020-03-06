import torch
import torch.nn.functional as F

def test_conv2d_1():
    kernel = torch.rand(8, 4, 3, 3)
    inputs = torch.rand(1, 4, 5, 5)

    kernel_hb = kernel.hammerblade()
    inputs_hb = inputs.hammerblade()

    conv_result_hb = F.conv2d(inputs_hb, kernel_hb, padding=1)
    conv_result = F.conv2d(inputs, kernel, padding=1)

    assert conv_result == conv_result_hb.cpu()
