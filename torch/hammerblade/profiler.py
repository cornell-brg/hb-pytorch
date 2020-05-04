import torch

def enable():
    torch._C._hammerblade_enable_profiler()

def disable():
    torch._C._hammerblade_disable_profiler()

def summary():
    return torch._C._hammerblade_summary_profiler()

def clear():
    torch._C._hammerblade_clear_profiler()
