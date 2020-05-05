import torch

class ProfilerRecord:

    def __init__(self, raw_entry):
        func, time_ms = raw_entry.split(";")
        self.func = func
        self.time_ms = float(time_ms)

    def calc_percentage(self, roi_time_ms):
        self.percentage = float(self.time_ms) / roi_time_ms * 100.0


class ProfilerROIRecord(ProfilerRecord):

    def __init__(self, raw_entry):
        super().__init__(raw_entry)
        assert self.func == "time_in_roi"
        self.func = "Total time in ROI"
        self.percentage = 100.0


class ProfilerAggTotalRecord(ProfilerRecord):

    def __init__(self, raw_entry):
        super().__init__(raw_entry)
        assert self.func == "agg_total"
        self.func = "Aggregate total"


class ProfilerOtherRecord(ProfilerRecord):

    def __init__(self, time_in_roi, agg_total):
        self.func = "Other"
        self.time_ms = time_in_roi - agg_total
        self.percentage = self.time_ms / time_in_roi * 100.0


class ProfilerTopLvlFuncRecord(ProfilerRecord):

    def __init__(self, raw_entry):
        super().__init__(raw_entry)
        func = self.func
        func = func.split("(")[0]
        func = func.split("::")[-1]
        func = "aten::" + func
        self.func = func


def enable():
    torch._C._aten_profiler_start()

def disable():
    torch._C._aten_profiler_end()

def add_beacon(signature):
    try:
        torch._C._aten_profiler_add_beacon(signature)
    except AttributeError:
        print("PyTorch is not built with profiling")

def clear_beacon():
    try:
        torch._C._aten_profiler_clear_beacon()
    except AttributeError:
        print("PyTorch is not built with profiling")

def raw_dump():
    try:
        return torch._C._aten_profiler_dump()
    except AttributeError:
        print("PyTorch is not built with profiling")

def stack_print():
    try:
        torch._C._aten_profiler_stack_print()
    except AttributeError:
        print("PyTorch is not built with profiling")

def unimpl_print():
    try:
        torch._C._aten_profiler_unimpl_print()
    except AttributeError:
        print("PyTorch is not built with profiling")

def _process_raw_data(raw_data=None):
    if raw_data is None:
        raw_data = torch._C._aten_profiler_dump()
    raw_entries = raw_data.splitlines()
    time_in_roi = ProfilerROIRecord(raw_entries[0])
    agg_total = ProfilerAggTotalRecord(raw_entries[-1])
    other = ProfilerOtherRecord(time_in_roi.time_ms, agg_total.time_ms)
    entries = []

    for e in raw_entries[1:-1]:
        entry = ProfilerTopLvlFuncRecord(e)
        entry.calc_percentage(time_in_roi.time_ms)
        entries.append(entry)
    entries.sort(key=lambda e: e.time_ms, reverse=True)
    entries += [other, time_in_roi]

    return entries

def fancy_print(raw_data=None):
    try:
        entries = _process_raw_data(raw_data)

        for e in entries:
            func = e.func
            time = e.time_ms / 1000.0
            percentage = e.percentage
            print('{func:30}     {time:.2f} {percentage:.1f}%'.format(
                func=func, time=time, percentage=percentage))
    except AttributeError:
        print("PyTorch is not built with profiling")

def fancy_print_to_file(raw_data=None, filename="profiling.txt"):
    try:
        entries = _process_raw_data(raw_data)

        with open(filename, "w") as file:
            for e in entries:
                func = e.func
                time = e.time_ms / 1000.0
                percentage = e.percentage
                file.write('{func:30}     {time:.2f} {percentage:.1f}%\n'.format(
                    func=func, time=time, percentage=percentage))
    except AttributeError:
        print("PyTorch is not built with profiling")

def latex_table(raw_data=None):
    try:
        entries = _process_raw_data(raw_data)

        header = "\\begin{table}[t]\n" \
                 "\\begin{tabular}{lrr}\n" \
                 "\\toprule\n" \
                 "& \\textbf{Time} & \\textbf{Percent} \\\\\n" \
                 "\\textbf{Kernel} & \\textbf{(s)} & \\textbf{of Total} \\\\ \\midrule\n"
        print(header)

        for e in entries:
            func = e.func
            func = func.replace("_", "\\_")
            time = e.time_ms / 1000.0
            percentage = e.percentage
            print('\\textbf{{{func:30}}} &  {time:.2f} & {percentage:.1f}\\% \\\\'.format(
                func=func, time=time, percentage=percentage))

        footer = "\\bottomrule\n" \
                 "\\end{tabular}\n" \
                 "\\label{tbl-plat}\n" \
                 "\\end{table}\n"
        print(footer)
    except AttributeError:
        print("PyTorch is not built with profiling")
