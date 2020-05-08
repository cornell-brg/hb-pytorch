import torch

# --------- torch.hb_profiler.chart APIs ---------

def add_beacon(signature):
    try:
        torch._C._hb_profiler_chart_add_beacon(signature)
    except AttributeError:
        print("PyTorch is not built with profiling")

def clear_beacon():
    try:
        torch._C._hb_profiler_chart_clear_beacon()
    except AttributeError:
        print("PyTorch is not built with profiling")

def print():
    try:
        return torch._C._hb_profiler_chart_print()
    except AttributeError:
        print("PyTorch is not built with profiling")
