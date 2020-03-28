import torch

def _hb_tensor_from_cpu_with_grad(input_t):
    """
    Returns a HB tensor with gradient enabled that's a leaf node of
    the computation graph.
    """
    return input_t.hammerblade().clone().detach().requires_grad_(True)
