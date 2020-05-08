import torch

class ProfilerStatus:

    def __init__(self):
        self.is_in_ROI = False


profiler_status = ProfilerStatus()

def enable():
    profiler_status.is_in_ROI = True
    torch._C._hb_profiler_start()

def disable():
    profiler_status.is_in_ROI = False
    torch._C._hb_profiler_end()

def is_in_ROI():
    return profiler_status.is_in_ROI

# --------- import components ---------
import torch.hb_profiler.exec_time
import torch.hb_profiler.unimpl
import torch.hb_profiler.chart
import torch.hb_profiler.route
