import torch
import torch.nn as nn

def init_hb_tensor(input_t):
    """
    Returns a HB tensor that's a leaf node of the computation graph.
    """
    return input_t.hammerblade().clone().detach().requires_grad_(
        input_t.requires_grad)

class CheckLayer(nn.Module):
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
                    "Forward must be called on CPU model first"
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
                    "Forward must be called on CPU model first"
            assert torch.allclose(CheckLayerFunction.cpu_bkwd, x.cpu(), atol=1e-7) is True, \
                    "hbutils.CheckLayer failed:\n" + \
                    "CPU output:\n" + str(CheckLayerFunction.cpu_bkwd) + "\n" \
                    "HB output:\n" + str(x) + "\n"
            CheckLayerFunction.cpu_bkwd = None
        else:
            CheckLayerFunction.cpu_bkwd = x.clone()

        return x

