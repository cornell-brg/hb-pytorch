import json
import re
import csv
import sys

ROUTE_JSON = 'sinkhorn_wmd.json'
HB_STATS = 'run_{}/manycore_stats.log'
CPU_LOG = 'cpu_run/log.txt'


def cycles_from_stats(stats):
    """Given the text contents of a `manycore_stats.log` file, extract the
    total number of cycles for the kernel execution.
    """
    lines = stats.splitlines()
    overview = lines[3]
    cycles = int(overview.split()[6])
    return cycles


def kernel_name(sig):
    """Given a C++ function signature, extract the base name, for human
    legibility.
    """
    return re.search(r'::(\w+)\(', sig).group(1)


def times_from_log(log):
    """Given a CPU execution log, look for the output from
    `hammerblade.profiler.stats` that breaks down the amount of wall-clock
    time spent in each kernel. Generate (kernel, time) pairs.
    """
    in_report = False
    for line in log.splitlines():
        if 'Kernel execution time' in line:
            in_report = True
            continue
        
        if in_report:
            kernel, tm, pct = line.strip().split()
            if kernel.startswith('aten::'):
                _, kernel = kernel.split('::')
            if 'time_in_roi' in kernel:
                break
            yield kernel, float(tm)


def collect():
    with open(ROUTE_JSON) as f:
        kernels = json.load(f)

    # Load all HB cycles statistics.
    hb_cycles = {}
    for i, kernel in enumerate(kernels):
        stats_fn = HB_STATS.format(i)
        with open(stats_fn) as f:
            stats_txt = f.read()
        hb_cycles[kernel_name(kernel['signature'])] = \
            cycles_from_stats(stats_txt)

    # Load CPU time breakdown.
    with open(CPU_LOG) as f:
        log_txt = f.read()
    cpu_times = dict(times_from_log(log_txt))

    # Dump a CSV.
    writer = csv.DictWriter(
        sys.stdout,
        ['kernel', 'cpu_time', 'hb_cycles']
    )
    writer.writeheader()
    for kernel in sorted(set(hb_cycles).union(cpu_times)):
        writer.writerow({
            'kernel': kernel,
            'cpu_time': cpu_times.get(kernel),
            'hb_cycles': hb_cycles.get(kernel),
        })


if __name__ == '__main__':
    collect()
