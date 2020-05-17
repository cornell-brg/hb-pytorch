import torch

# --------- import components ---------
import torch.hammerblade.profiler.exec_time
import torch.hammerblade.profiler.unimpl
import torch.hammerblade.profiler.chart
import torch.hammerblade.profiler.route

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

def stats(key=None, trimming=False):
    buffer = ""
    if key is None:
        key = ['ExecTime', 'Unimpl']
    for k in key:
        if k == 'ExecTime':
            buffer += "Kernel execution time:\n"
            buffer += exec_time.fancy_print(trimming)
            buffer += "\n"
            buffer += "------------------------------------------------------------\n"
            buffer += "\n"
        elif k == 'ExecTime-Latex':
            buffer += "Kernel execution time: (latex table)\n"
            buffer += exec_time.latex_table(trimming)
            buffer += "\n"
            buffer += "------------------------------------------------------------\n"
            buffer += "\n"
        elif k == 'ExecTime-Raw':
            buffer += "Kernel execution time raw stack:\n"
            buffer += exec_time.raw_stack()
            buffer += "\n"
            buffer += "------------------------------------------------------------\n"
            buffer += "\n"
        elif k == 'Unimpl':
            buffer += "Kernels used on CPU but not yet implemented on HB:\n"
            buffer += unimpl.fancy_print()
            buffer += "\n"
            buffer += "------------------------------------------------------------\n"
            buffer += "\n"
        else:
            print("Unknown stat key -- " + k)
            print("available ones are: ExecTime, ExecTime-Latex, ExecTime-Raw, Unimpl")
            return ""
    return buffer
