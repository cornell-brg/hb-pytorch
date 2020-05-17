import torch

# --------- torch.hb_profiler.unimpl APIs ---------

def fancy_print():
    try:
        return torch._C._hb_profiler_unimpl_print()
    except AttributeError:
        print("PyTorch is not built with profiling")
