import json
import re
import csv
import sys

from calc_7nm import energy_7nm

ROUTE_JSON = 'sinkhorn_wmd.json'
HB_STATS = 'run_{}/manycore_stats.log'
HB_LOG = 'run_{}/log.txt'
CPU_LOG = 'cpu_run/log.txt'

HB_FREQ = 10 ** 9  # 1 GHz.
HB_MACHINE_FRAC = 16  # Simulating 1/16th of the machine.
HB_DATA_FRAC = 16  # Used this fraction of the CPU's data.

CPU_TDP = 165  # Xeon power (watts).


def cycles_from_stats(stats):
    """Given the text contents of a `manycore_stats.log` file, extract the
    total number of cycles for the kernel execution.
    """
    lines = stats.splitlines()
    for line in lines:
        if line.startswith('kernel'):
            return int(line.split()[6])


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


def parse_tree(log):
    """Given an execution log from any run, look for the "tree" output
    from `hammerblade.profiler.exec_time.raw_stack` and parse it into
    (level, function, time) tuples.
    """
    for line in log.splitlines():
        if line.strip().startswith('|- Node'):
            indent, rest = line.split('|-', 1)
            level = len(indent) // 2
            match = re.search(r'Node\((.*) : (\d+\.\d+)\)', line)
            sig, tm = match.groups()

            micros = int(tm.split('.')[0])  # Data reported in microseconds.

            yield level, sig, micros / 10**6


def total_times_from_tree(log):
    """Given an execution log from any run, use the `raw_stack` tree to
    get the total execution time for top-level kernel invocations.
    Generate (kernel, time) pairs.
    """
    for level, sig, secs in parse_tree(log):
        if level == 1:
            kernel = kernel_name(sig) if sig.startswith('at::') else sig
            yield kernel, secs


def trimmed_times_from_tree(log, marker='@HB_LOG@'):
    cur_kernel = None
    cur_total = None
    in_marker = False
    for level, sig, secs in parse_tree(log):
        # Check for a new top-level kernel.
        if level == 1:
            if cur_kernel is not None:
                yield cur_kernel, cur_total
            cur_kernel = kernel_name(sig) if sig.startswith('at::') else sig
            if marker:
                cur_total = None
                in_marker = False
            else:
                cur_total = secs
            continue

        if level == 2 and sig == marker:
            in_marker = True
            cur_total = secs
            continue
        if level == 2:
            in_marker = False
            continue
        if not in_marker:
            continue

        # Check for trimmed functions.
        if '@BSG_API_CALL@' in sig or '@OFFLOAD_KERNEL@' in sig:
            cur_total -= secs  # Trim this time!

    # Emit final kernel.
    yield cur_kernel, cur_total


def hb_cycles_to_time(cycles):
    """Compute seconds from simulation cycles, assuming weak scaling and the
    hardware frequence.
    """
    # The scale factor is the amount slower we would run if we used *all* the
    # data. If we were simulating the *full* machine, this would just be
    # `HB_DATA_FRAC`, i.e., how much *less* data HB had to process than the
    # CPU. However, we simulate only `HB_MACHINE_FRAC` of the machine and
    # assume that expanding the data by *that* amount would lead to the same
    # execution time.
    scale_factor = HB_DATA_FRAC / HB_MACHINE_FRAC

    # The amount of "real" time we simulated for, according to the machine
    # frequency.
    sim_secs = cycles / HB_FREQ

    return sim_secs * scale_factor


def collect(summary):
    with open(ROUTE_JSON) as f:
        kernels = json.load(f)

    # Load results from every HB run (one per kernel).
    hb_cycles = {}
    hb_host_times = {}
    hb_energies = {}
    for i, kernel in enumerate(kernels):
        kname = kernel_name(kernel['signature'])
        # Load HB cycles from statistics dump.
        stats_fn = HB_STATS.format(i)
        with open(stats_fn) as f:
            stats_txt = f.read()
        hb_cycles[kname] = cycles_from_stats(stats_txt)

        # Load host-side times from the log.
        log_fn = HB_LOG.format(i)
        with open(log_fn) as f:
            log_txt = f.read()
        trimmed_times = dict(trimmed_times_from_tree(log_txt))
        hb_host_times[kname] = trimmed_times[kname]

        # Run energy model, convert to joules.
        energy = energy_7nm(stats_txt) * HB_DATA_FRAC
        hb_energies[kname] = energy / 10**6

    # Load CPU time breakdown.
    with open(CPU_LOG) as f:
        log_txt = f.read()
    cpu_times = list(total_times_from_tree(log_txt))
    assert set(k for k, _ in cpu_times).issuperset(hb_cycles)

    if summary:
        # Summarize a few overall results.
        cpu_total_time = 0.0
        hb_total_time = 0.0
        hb_kern_energy = 0.0
        cpu_kern_energy = 0.0
        for kernel, cpu_time in cpu_times:
            cpu_total_time += cpu_time
            if kernel in hb_cycles:
                hb_total_time += hb_cycles_to_time(hb_cycles[kernel]) \
                    + hb_host_times[kernel]
                hb_kern_energy += hb_energies[kernel]
                cpu_kern_energy += cpu_time * CPU_TDP
            else:
                # Where we don't have a kernel, charge us for CPU execution.
                hb_total_time += cpu_time

        print('CPU time: {:.3f} s'.format(cpu_total_time))
        print('CPU+HB time: {:.3f} s'.format(hb_total_time))
        print('Speedup: {:.1f}x'.format(cpu_total_time / hb_total_time))
        print('CPU kernel energy: {:.1f} J'.format(cpu_kern_energy))
        print('HB kernel energy: {:.1f} mJ'.format(hb_kern_energy * 10**3))
        print('Energy ratio: {:.0f}x'.format(cpu_kern_energy / hb_kern_energy))

    else:
        # Dump a CSV.
        writer = csv.DictWriter(
            sys.stdout,
            ['kernel', 'cpu_time', 'cpu_energy', 'hb_cycles', 'hb_time',
             'hb_host_time', 'hb_energy']
        )
        writer.writeheader()
        for kernel, cpu_time in cpu_times:
            writer.writerow({
                'kernel': kernel,
                'cpu_time': cpu_time,
                'cpu_energy': cpu_time * CPU_TDP,
                'hb_cycles': hb_cycles.get(kernel),
                'hb_time': (hb_cycles_to_time(hb_cycles[kernel])
                            if kernel in hb_cycles else ''),
                'hb_host_time': (hb_host_times[kernel]
                                 if kernel in hb_host_times else ''),
                'hb_energy': (hb_energies[kernel]
                              if kernel in hb_energies else ''),
            })


if __name__ == '__main__':
    collect('-s' in sys.argv)
