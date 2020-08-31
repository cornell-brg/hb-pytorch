import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import csv

font = {'family' : 'normal',
        'weight' : 'regular',
        'size'   : 20}

matplotlib.rc('font', **font)

# ---------------------------------------------------------
# Parse aggregated data
# ---------------------------------------------------------

class DataPoint:
    def __init__(self, name, numpy_xeon, pytorch_xeon, host, device):
        self.name = name
        self.numpy_xeon = numpy_xeon
        self.pytorch_xeon = pytorch_xeon
        self.host = host
        self.device = device
    def __str__(self):
        return "| {func:<22} |{numpy_xeon:>16} |{pytorch_xeon:>16} |{host:>16} |{device:>18} |".format(
                func = self.name,
                numpy_xeon = self.numpy_xeon,
                pytorch_xeon = self.pytorch_xeon,
                host = self.host,
                device = self.device)


def parse_table(filename):
    with open(filename, "r") as f:
        data = f.readlines()
    kernels = []
    for d in data[3:]:
        if d.startswith("+"):
            continue
        d = d.split("|")
        print(d)
        name = "".join(d[1].split())

        # Skip non-kernels for plot.
        if name in ('total', 'data_transfer'):
            continue

        # Skip insignificant kernels for plot.
        if name in ('zeros', 'clone'):
            continue

        numpy_xeon = float(d[2]) / 10**3
        pytorch_xeon = float(d[3]) / 10**3
        host = float(d[4]) / 10**3
        device = float(d[5]) / 10**3
        kernels.append(DataPoint(name, numpy_xeon, pytorch_xeon, host, device))
        print(kernels[-1])
    return kernels


def parse_csv(filename):
    with open(filename) as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row['hb_time']:
                yield DataPoint(
                    row['kernel'],
                    0.0,  # No NumPy.
                    float(row['cpu_time']),
                    float(row['hb_host_time']),
                    float(row['hb_time']),
                )

# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

def plot_bar(kernels, yticks, legend, title, outfile):
    N = len(kernels)
    print(N)
    numpy_xeon_time = [k.numpy_xeon for k in kernels]
    pytorch_xeon_time = [k.pytorch_xeon for k in kernels]
    host_time = [k.host for k in kernels]
    device_time = [k.device for k in kernels]

    plt.figure(figsize=(2 * N, 7), dpi=80)

    ax = plt.subplot(1,1,1)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ind = np.arange(N)
    print(ind)
    width = 0.3
    if any(numpy_xeon_time):
        plt.bar(ind - width, numpy_xeon_time, width, align='center',
                label='Xeon + NumPy', edgecolor='black', color='#003f5c')
    plt.bar(ind, pytorch_xeon_time, width, align='center',
            label='Xeon + PyTorch', edgecolor='black', color='#58508d')
    plt.bar(ind + width, device_time, width, align='center',
            label='HammerBlade + PyTorch\n2048 cores @ 1 GHz',
            edgecolor='black', color='#bc5090')
    plt.bar(ind + width, host_time, width, align='center',
            bottom=device_time, label='HammerBlade Host',
            edgecolor='black', color='#ffa600')

    # for i in range(len(kernels)):
    #     plt.text(ind[i] - 0.20, xeon_time[i] + 0.05, ("{:3.1f}".format(xeon_time[i])))

#    plt.xlabel('Numpy and Pytorch Tensor Operators Used in Lgc-ista', fontsize=15)
    plt.ylabel('Execution Time (s)', fontsize=15)
    plt.title(title, fontsize=17, fontweight='bold')
    plt.ylim(0, max(yticks))
#    plt.text(pytorch_xeon_time[])

    ylabel = np.array(yticks)

    plt.xticks(ind, [k.name for k in kernels], rotation=70, fontsize=15)
    plt.yticks(ylabel, fontsize=15)
#    plt.yscale('log')
    if legend:
        plt.legend(loc='best', frameon=True, fancybox=False, framealpha=1,
                   labelspacing=0.7, fontsize=15)
    plt.tight_layout()
    plt.savefig(outfile)


if __name__ == "__main__":
    lgc_data = parse_table("august.txt")
    plot_bar(lgc_data, (0, 5, 10, 15, 20, 25), True,
             'LGC-ISTA', 'lgc.pdf')

    swmd_data = list(parse_csv('results.csv'))
    plot_bar(swmd_data, (0.00, 0.02, 0.04, 0.06, 0.08, 0.10), False,
             'Sinkhorn WMD', 'swmd.pdf')
