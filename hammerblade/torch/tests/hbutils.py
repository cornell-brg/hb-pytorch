import torch

def init_hb_tensor(input_t):
    """
    Returns a HB tensor that's a leaf node of the computation graph.
    """
    return input_t.hammerblade().clone().detach().requires_grad_(
            input_t.requires_grad)
