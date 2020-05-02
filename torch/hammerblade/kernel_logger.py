import torch

def enable():
    torch._C._hammerblade_enable_kernel_call_logger()

def disable():
    torch._C._hammerblade_disable_kernel_call_logger()

def json():
    return torch._C._hammerblade_log_kernel_call_logger()

def clear():
    torch._C._hammerblade_clear_kernel_call_logger()
