import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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
        numpy_xeon = float(d[2])
        pytorch_xeon = float(d[3])
        host = float(d[4])
        device = float(d[5])
        kernels.append(DataPoint(name, numpy_xeon, pytorch_xeon, host, device))
        print(kernels[-1])
    return kernels


# ---------------------------------------------------------
# Plot
# ---------------------------------------------------------

def plot_bar(kernels):

    N = len(kernels)
    print(N)
    numpy_xeon_time = [k.numpy_xeon for k in kernels]
    pytorch_xeon_time = [k.pytorch_xeon for k in kernels]
    host_time = [k.host for k in kernels]
    device_time = [k.device for k in kernels]

    plt.figure(figsize=(13,7), dpi=80)

    ax = plt.subplot(1,1,1)
    ax.spines['bottom'].set_linewidth(3)
    ax.spines['left'].set_linewidth(3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    ind = np.arange(N)
    print(ind)
    width = 0.3
    plt.bar(ind - width, numpy_xeon_time, width, align='center', label='Intel Xeon Gold 6242\n16C/32T @2.8GHz w/ 6xDDR4\nNumPy Parallel Version w/ BLAS', edgecolor='black')
    plt.bar(ind, pytorch_xeon_time, width, align='center', label='Intel Xeon Gold 6242\n16C/32T @2.8GHz w/ 6xDDR4\nPyTorch Parallel Version w/ BLAS', edgecolor='black')
    plt.bar(ind + width, device_time, width, align='center', label='HammerBlade\n2K cores @ 1 GHz w/ 32xHBM2\nPyTorch HB/CUDA-Lite Port', edgecolor='black')
    plt.bar(ind + width, host_time, width, align='center', bottom=device_time, label='HammerBlade Host', edgecolor='black')

    # for i in range(len(kernels)):
    #     plt.text(ind[i] - 0.20, xeon_time[i] + 0.05, ("{:3.1f}".format(xeon_time[i])))

#    plt.xlabel('Numpy and Pytorch Tensor Operators Used in Lgc-ista', fontsize=15)
    plt.ylabel('Execution Time (ms)', fontsize=15)
    plt.title('Comparison of Intel Xeon vs. HammerBlade on SDH Sparse Workload lgc-ista', fontsize=17, fontweight='bold')
    plt.ylim(0, 25000)
#    plt.text(pytorch_xeon_time[])

    ylabel = np.array([0, 5000, 10000, 15000, 20000, 25000])

    plt.xticks(ind, [k.name for k in kernels], rotation=70, fontsize=15)
    plt.yticks(ylabel, fontsize=15)
    plt.hlines(ylabel, xmin=0, xmax=9, colors='grey', linestyles='dashed')
#    plt.yscale('log')
    plt.legend(loc='best', frameon=True, fancybox=False, framealpha=1, labelspacing=0.7, fontsize=15)
    plt.tight_layout()
    plt.savefig('results.pdf')


if __name__ == "__main__":
    lgc_data = parse_table("august.txt")
    plot_bar(lgc_data)
