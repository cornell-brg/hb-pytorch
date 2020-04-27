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

def raw_dump():
    return torch._C._aten_profiler_dump()

def fancy_print(raw_data=None):
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
    entries += [other, time_in_roi]

    for e in entries:
        func = e.func
        time = e.time_ms / 1000.0
        percentage = e.percentage
        print(f'{func:30}     {time:.2f} {percentage:.1f}%')
