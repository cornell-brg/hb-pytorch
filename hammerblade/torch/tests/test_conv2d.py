import torch
import torch.nn.functional as F
import os
import pytest

def _test_conv2d(inputs, kernel, padding=1, stride=1):
    inputs_hb = inputs.hammerblade()
    kernel_hb = kernel.hammerblade()

    conv_result_hb = F.conv2d(inputs_hb, kernel_hb,
                              padding=padding, stride=stride)
    conv_result = F.conv2d(inputs, kernel,
                           padding=padding, stride=stride)

    assert torch.allclose(conv_result, conv_result_hb.cpu())

def test_conv2d_1():
    """
    Single batch, single channel
    """
    kernel = torch.rand(1, 1, 3, 3)
    inputs = torch.rand(1, 1, 5, 5)

    _test_conv2d(inputs, kernel)

def test_conv2d_2():
    """
    Single batch, multi-channel
    """
    kernel = torch.rand(1, 2, 3, 3)
    inputs = torch.rand(1, 2, 5, 5)

    _test_conv2d(inputs, kernel)

def test_conv2d_3():
    """
    Multi-batch, single channel
    """
    kernel = torch.rand(2, 1, 3, 3)
    inputs = torch.rand(1, 1, 5, 5)

    _test_conv2d(inputs, kernel)

def test_conv2d_4():
    """
    Multi-batch, multi-channel
    """
    kernel = torch.rand(5, 3, 3, 3)
    inputs = torch.rand(2, 3, 5, 5)
    padding = (1, 2)
    stride = (1, 2)

    _test_conv2d(inputs, kernel, padding, stride)

@pytest.mark.skipif(os.environ.get('USE_HB_EMUL') == None, reason="Prohibitively slow on cosim")
def test_conv2d_batch_input_output():
    """
    Combinations of batch, input and output channel sizes
    """
    width = 5
    height = 5
    kernel_size = 3

    for batch_size in range(1, 5):
        for input_channels in range(1, 5):
            for output_channels in range(1, 5):
                inputs = torch.rand(batch_size, input_channels, width, height)
                kernel = torch.rand(output_channels, input_channels, kernel_size,
                                    kernel_size)
                _test_conv2d(inputs, kernel)

@pytest.mark.skipif(os.environ.get('USE_HB_EMUL') == None, reason="Prohibitively slow on cosim")
def test_conv2d_width_height_kernel():
    """
    Combinations of width, height and kernel_size
    """
    batch_size = 2
    input_channels = 2
    output_channels = 4

    for width in range(8, 16):
        for height in range(8, 16):
            for kernel_size in range(1, 5):
                inputs = torch.rand(batch_size, input_channels, width, height)
                kernel = torch.rand(output_channels, input_channels, kernel_size,
                                    kernel_size)
                _test_conv2d(inputs, kernel)

@pytest.mark.skipif(os.environ.get('USE_HB_EMUL') == None, reason="Prohibitively slow on cosim")
def test_conv2d_width_height_kernel_pad_stride():
    """
    Combinations of width, height, kernel_size, padding and stride
    """
    batch_size = 2
    input_channels = 2
    output_channels = 4

    for width in range(8, 16):
        for height in range(8, 16):
            for kernel_size in range(1, 5):
                for pad in range(1, kernel_size):
                    for stride in range(1, kernel_size):
                        inputs = torch.rand(batch_size, input_channels, width,
                                            height)
                        kernel = torch.rand(output_channels, input_channels,
                                            kernel_size, kernel_size)
                        _test_conv2d(inputs, kernel)
