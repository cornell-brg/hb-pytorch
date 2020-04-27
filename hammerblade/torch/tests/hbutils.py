"""
HB PyTorch utilities
04/02/2020 Bandhav Veluri
"""

import torch
import torch.nn as nn

def init_hb_tensor(input_t):
    """
    Returns a HB tensor that's a leaf node of the computation graph.
    """
    return input_t.hammerblade().clone().detach().requires_grad_(
        input_t.requires_grad)

class CheckLayer(nn.Module):
    """
    Debug layer to cross-check the flow of forward values and backward
    gradients with those of CPU.

    Usage:

    Following exmaple checks the forward of `Conv2d` layer and backward of
    second `ReLU` layer:

        class MyNet(nn.Module):
            def __init__(self):
                super(MyNet, self).__init__()

                self.conv = nn.Sequential(
                    nn.Conv2d(1, 1, kernel_size=(2,2)),
                    ###
                    hbutils.CheckLayer(),
                    ###
                    nn.ReLU(),
                    nn.LogSoftmax(dim=-1),
                )

            def forward(self, x):
                x = self.conv(x)
                return x

        net = MyNet()
        net_hb = MyNet().hammerblade()
        net_hb.load_state_dict(net.state_dict())

        inputs = torch.rand(1, 1, 4, 4, requires_grad=True)
        inputs_hb = hbutils.init_hb_tensor(inputs)

        output = net(inputs) # CheckLayer registers output of CPU `Conv2d` layer
        output_hb = net_hb(inputs_hb)  # CheckLayer checks output of HB `Conv2d` layer

        grad = torch.rand(output.shape)
        grad_hb = grad.hammerblade()

        output.backward(grad) # Registers gradient of second `ReLU` layer
        output_hb.backward(grad_hb) # Checks gradient of corrsponding HB layer
    """
    def __init__(self):
        super(CheckLayer, self).__init__()

    def forward(self, x):
        return CheckLayerFunction.apply(x)

class CheckLayerFunction(torch.autograd.Function):
    cpu_fwd = None
    cpu_bkwd = None

    @staticmethod
    def forward(ctx, x):
        if x.is_hammerblade:
            assert CheckLayerFunction.cpu_fwd is not None, \
                "Forward must be called on the CPU model first"
            assert torch.allclose(CheckLayerFunction.cpu_fwd, x.cpu(), atol=1e-7) is True, \
                "hbutils.CheckLayer failed:\n" + \
                "CPU output:\n" + str(CheckLayerFunction.cpu_fwd) + "\n" \
                "HB output:\n" + str(x) + "\n"
            CheckLayerFunction.cpu_fwd = None
        else:
            CheckLayerFunction.cpu_fwd = x.clone()

        return x

    @staticmethod
    def backward(ctx, x):
        if x.is_hammerblade:
            assert CheckLayerFunction.cpu_bkwd is not None, \
                "Backward must be called on the CPU model first"
            assert torch.allclose(CheckLayerFunction.cpu_bkwd, x.cpu(), atol=1e-7) is True, \
                "hbutils.CheckLayer failed:\n" + \
                "CPU gradient:\n" + str(CheckLayerFunction.cpu_bkwd) + "\n" \
                "HB gradient:\n" + str(x) + "\n"
            CheckLayerFunction.cpu_bkwd = None
        else:
            CheckLayerFunction.cpu_bkwd = x.clone()

        return x
