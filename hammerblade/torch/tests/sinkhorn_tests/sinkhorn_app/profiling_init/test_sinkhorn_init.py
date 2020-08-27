import numpy
import scipy.sparse
import os
import sys
import torch
import json
from time import time

# Data files. (Ask Adrian for these.)
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_MAT = os.path.join(DATA_DIR, 'cache-mat.npz')
DATA_VECS = os.path.join(DATA_DIR, 'cache-vecs.npy')

# Kernel parameters.
N_FRACTION = 16*16  # use N_DOCS/N_FRACTION of the data
N_DOCS = int(4096 / N_FRACTION)
Q_FRACTION = 20  # use QUERY_IDX/Q_FRACTION of the data
QUERY_IDX = int(100 / Q_FRACTION)  # Was 100; lowered to allow even smaller runs.
LAMBDA = 1

start_time = None
def begin_profile(on_hb):
    if (not on_hb):
        start_time = time()
    torch.hammerblade.profiler.enable()

def end_profile(on_hb, mult=None):
    torch.hammerblade.profiler.disable()
    if (not on_hb):
        end_time = time()
        elapsed = end_time - start_time
        print("elapsed time:", elapsed)
        if mult:
            print("elapsed time *",mult,":", elapsed*mult)

def load_data():
    """Load data for the Sinkhorn WMD kernel.
    """
    # Load data.
    vecs = numpy.load(DATA_VECS)
    mat = scipy.sparse.load_npz(DATA_MAT)
    print("vecs size:", vecs.shape)
    mat = mat[:, :N_DOCS]  # Use a subset of the data.
    print("mat shape:", mat.shape)
    # The query vector.
    r = numpy.asarray(mat[:, QUERY_IDX].todense()).squeeze()

    # mat could theoretically be stored as its transpose, so don't count 
    matT = mat.T

    # Convert arrays to PyTorch tensors.
    r = torch.FloatTensor(r)
    cT_coo = matT.tocoo()
    cT = torch.sparse.FloatTensor(
        torch.LongTensor(numpy.vstack((cT_coo.row, cT_coo.col))),
        torch.FloatTensor(cT_coo.data),
        torch.Size(cT_coo.shape),
    )

    vecs = torch.FloatTensor(vecs)

    return r, cT, vecs


def sinkhorn_test():
    # Use `--hb` to run in HammerBlade mode. Otherwise, we run all native.
    on_hb = '--hb' in sys.argv

    if on_hb:
        torch.hammerblade.init()

        # Set up HammerBlade "routing," which tells kernels to run on HB
        # instead of on the CPU.
        with open('sinkhorn_wmd.json') as f:
            route_data = json.load(f)
        for kernel in route_data:
            print('offloading kernel', kernel['signature'])
            kernel['offload'] = True
        torch.hammerblade.profiler.route.set_route_from_json(route_data)

    # Load data and run the kernel.
    print('loading data')

    begin_profile(on_hb)
    r, cT, vecs = load_data()
    end_profile(on_hb, Q_FRACTION)

    print('done loading data')

    # Dump profiling results.
    print(torch.hammerblade.profiler.stats())
    print("done")


# torch.hammerblade.profiler.chart.add("at::Tensor at::SparseCPUType::{anonymous}::dstmm(const at::Tensor&, const at::Tensor&)")
# torch.hammerblade.profiler.chart.add("at::Tensor at::SparseCPUType::{anonymous}::dstmmt(const at::Tensor&, const at::Tensor&)")
# torch.hammerblade.profiler.chart.add("at::Tensor at::SparseCPUType::{anonymous}::sddtmm(const at::Tensor&, const at::Tensor&, const at::Tensor&)")

# print(torch.hammerblade.profiler.exec_time.raw_stack())
# print(torch.hammerblade.profiler.chart.json())

if __name__ == '__main__':
    sinkhorn_test()