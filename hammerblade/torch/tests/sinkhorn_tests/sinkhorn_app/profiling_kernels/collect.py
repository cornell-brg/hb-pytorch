import json
import re

ROUTE_JSON = 'sinkhorn_wmd.json'
HB_STATS = 'run_{}/manycore_stats.log'


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

    print(hb_cycles)


if __name__ == '__main__':
    collect()
