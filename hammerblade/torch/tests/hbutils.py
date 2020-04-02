import torch
import torch.nn as nn

def init_hb_tensor(input_t):
    """
    Returns a HB tensor that's a leaf node of the computation graph.
    """
    return input_t.hammerblade().clone().detach().requires_grad_(
        input_t.requires_grad)

class CheckLayer(nn.Module):
    fwd = None

    def __init__(self):
        super(CheckLayer, self).__init__()

    def forward(self, x):
        with torch.no_grad():
            if x.is_hammerblade:
                assert CheckLayer.fwd is not None, "Forward must be called on CPU model first"
                assert torch.allclose(CheckLayer.fwd, x.cpu(), atol=1e-7) is True, \
                        "hbutils.CheckLayer failed:\n" + \
                        "CPU output:\n" + str(CheckLayer.fwd) + "\n"\
                        "HB output:\n" + str(x) + "\n"
            else:
                CheckLayer.fwd = x.clone()

        return x
