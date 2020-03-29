"""
Unit tests for conv2d operator
03/11/2020 Bandhav Veluri
"""
import torch
import torch.nn.functional as F
import os
import pytest

def _test_conv2d(inputs, kernel, padding=1, stride=1, bias=None):
    inputs_hb = inputs.hammerblade()
    kernel_hb = kernel.hammerblade()
    bias_hb = None if bias is None else bias.hammerblade()

    conv_result_hb = F.conv2d(inputs_hb, kernel_hb,
                              padding=padding, stride=stride, bias=bias_hb)
    conv_result = F.conv2d(inputs, kernel,
                           padding=padding, stride=stride, bias=bias)

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
    kernel = torch.rand(3, 3, 3, 3)
    inputs = torch.rand(2, 3, 5, 5)
    padding = (1, 2)
    stride = (1, 2)

    _test_conv2d(inputs, kernel, padding, stride)

def test_conv2d_5():
    """
    Multiple pads
    """
    kernel = torch.rand(1, 3, 3, 3)
    inputs = torch.rand(1, 3, 5, 5)
    padding = 2
    stride = 1

    _test_conv2d(inputs, kernel, padding, stride)

def test_conv2d_6():
    """
    Multiple pads and strides
    """
    kernel = torch.rand(2, 3, 3, 3)
    inputs = torch.rand(2, 3, 5, 5)
    padding = 2
    stride = 2

    _test_conv2d(inputs, kernel, padding, stride)

def test_conv2d_7():
    """
    Kernel size equal to image size
    """
    kernel = torch.rand(2, 3, 5, 5)
    inputs = torch.rand(2, 3, 5, 5)
    padding = 1
    stride = 1

    _test_conv2d(inputs, kernel, padding, stride)

def test_conv2d_8():
    """
    Large padding
    """
    kernel = torch.rand(2, 3, 3, 3)
    inputs = torch.rand(2, 3, 5, 5)
    padding = 3
    stride = 3

    _test_conv2d(inputs, kernel, padding, stride)

def test_conv2d_bias_1():
    """
    Conv2d bias single output channel
    """
    kernel = torch.rand(1, 1, 3, 3)
    inputs = torch.rand(1, 1, 5, 5)
    bias = torch.rand(1)

    _test_conv2d(inputs, kernel, bias=bias)

def test_conv2d_bias_2():
    """
    Conv2d bias multi-output channel
    """
    kernel = torch.rand(3, 1, 3, 3)
    inputs = torch.rand(1, 1, 5, 5)
    bias = torch.rand(3)

    _test_conv2d(inputs, kernel, bias=bias)

def test_conv2d_bias_3():
    """
    Conv2d bias multi-input multi-output channel
    """
    kernel = torch.rand(3, 2, 3, 3)
    inputs = torch.rand(1, 2, 5, 5)
    bias = torch.rand(3)

    _test_conv2d(inputs, kernel, bias=bias)

def test_conv2d_bias_4():
    """
    Conv2d bias with striding and padding
    """
    kernel = torch.rand(3, 2, 3, 3)
    inputs = torch.rand(1, 2, 5, 5)
    padding = (2, 3)
    stride = (1, 2)
    bias = torch.rand(3)

    _test_conv2d(inputs, kernel, padding, stride, bias)

@pytest.mark.skipif(os.environ.get('USE_HB_EMUL') is None, reason="Prohibitively slow on cosim")
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

@pytest.mark.skipif(os.environ.get('USE_HB_EMUL') is None, reason="Prohibitively slow on cosim")
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

@pytest.mark.skipif(os.environ.get('USE_HB_EMUL') is None, reason="Prohibitively slow on cosim")
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
