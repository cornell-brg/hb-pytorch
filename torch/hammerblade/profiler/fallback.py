import torch

# --------- torch.hb_profiler.fallback APIs ---------

def enable():
    try:
        torch._C._hb_profiler_fallback_enable()
    except AttributeError:
        print("PyTorch is not built with profiling")

def disable():
    try:
        torch._C._hb_profiler_fallback_disable()
    except AttributeError:
        print("PyTorch is not built with profiling")

def is_enabled():
    try:
        return torch._C._hb_profiler_fallback_is_enabled()
    except AttributeError:
        print("PyTorch is not built with profiling")
