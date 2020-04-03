import torch
import torch.nn as nn

def init_hb_tensor(input_t):
    """
    Returns a HB tensor that's a leaf node of the computation graph.
    """
    return input_t.hammerblade().clone().detach().requires_grad_(
        input_t.requires_grad)

class CheckLayerFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, cpu_fwd):
        if x.is_hammerblade:
            print("Running HB")
            assert cpu_fwd is not None, \
                    "Forward must be called on CPU model first"
            assert torch.allclose(cpu_fwd, x.cpu(), atol=1e-7) is True, \
                    "hbutils.CheckLayer failed:\n" + \
                    "CPU output:\n" + str(cpu_fwd) + "\n" \
                    "HB output:\n" + str(x) + "\n"
        else:
            print("Running CPU")
            cpu_fwd = x.clone()

        return x, cpu_fwd

    @staticmethod
    def backward(ctx, x, cpu_fwd):
        return x, cpu_fwd

class CheckLayer(nn.Module):
    cpu_fwd = None
    cpu_bkwd = None

    def __init__(self):
        super(CheckLayer, self).__init__()

    def forward(self, x):
        x, CheckLayer.cpu_fwd = CheckLayerFunction.apply(x, CheckLayer.cpu_fwd)
        if x.is_hammerblade:
            CheckLayer.cpu_fwd = None
        return x
