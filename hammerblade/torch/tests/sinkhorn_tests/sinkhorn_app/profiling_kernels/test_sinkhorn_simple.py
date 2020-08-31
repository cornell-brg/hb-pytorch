import numpy
import scipy.sparse
import os
import sys
import torch
import json
from time import time

# Kernel parameters.
TOTAL_DOCS = 4096
QUERY_IDX = 3  # Was 100; lowered to allow even smaller runs.
LAMBDA = 1

# Data files. (Ask Adrian for these.)
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
DATA_MAT = os.path.join(DATA_DIR, 'cache-mat.npz')
DATA_VECS = os.path.join(DATA_DIR, 'cache-vecs.npy')

# Kernel "routing" file.
ROUTE_JSON = os.path.join(os.path.dirname(__file__), 'sinkhorn_wmd.json')
# Kernel parameters.
HB_DATA_FRAC = 16 # fraction of data to use on hb, i.e. 1/(this value)

LAMBDA = 1
N_ITERS = 1

SAVE_FILE = '' #'scores.out'

def begin_profile(on_hb):
    start_time = None
    if not on_hb:
        start_time = time()
    torch.hammerblade.profiler.enable()
    return start_time

def end_profile(on_hb, start_time):
    torch.hammerblade.profiler.disable()
    if start_time:
        end_time = time()
        elapsed = end_time - start_time
        print("elapsed time:", elapsed)

def swmd_torch(r, cT, vecs, niters):
    """The actual Sinkhorn WMD kernel.
    """
    # I=(r > 0)
    sel = r > 0

    # r=r(I)
    r = r[sel].reshape(-1, 1)

    # M=M(I,:)
    M = torch.cdist(vecs[sel], vecs)

    # x=ones(length(r), size(c,2)) / length(r)
    a_dim = r.shape[0]
    b_nobs = cT.shape[0]
    xT = torch.ones((b_nobs, a_dim)) / a_dim

    # K=exp(-lambda * M)
    K = torch.exp(- M * LAMBDA)
    K_div_r = K / r
    K_T = K.T

    for it in range(niters):
        print('starting iteration {}'.format(it))

        uT = 1.0 / xT

        # Interesting property: sddtmmt(a,b,c) = sddtmm(a.T,c,b)
        # Compute `c * 1/(K_T @ u)` using a hand-rolled SDDMM.
        # v = c * (1.0 / _sddmm(c, K_T, u))
        # v = c * (1.0 / torch.sddtmm(c, K_T, uT)
        # vT = cT * 1.0 / torch.sddtmm(cT, uT, K_T)
        vT = cT * torch.sreciprocal_(torch.sddtmm(cT, uT, K_T))
        
        # custom dstmm.t():
        # x = _dsmp(K_div_r, v)
        # x = torch.dstmm(K_div_r, vT)
        xT = torch.dstmmt(K_div_r, vT)

    #Note: M is huge compared to uT, so use the sum(axis=0) instead of sum(axis=1) line
    # out = (uT * (vT @ (K_T * M.t())).sum(axis=1) 
    out = (uT.t() * torch.dstmm(K * M, vT)).sum(axis=0)
    return out


def load_data(n_docs):
    """Load data for the Sinkhorn WMD kernel.
    """
    # Load data.
    vecs = numpy.load(DATA_VECS)
    print("vecs size:", vecs.shape)
    mat = scipy.sparse.load_npz(DATA_MAT)
    mat = mat[:, :n_docs]  # Use a subset of the data.
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
    ).coalesce()

    vecs = torch.FloatTensor(vecs)

    return r, cT, vecs


def sinkhorn_test():
    # Use `--hb` to run in HammerBlade mode. Otherwise, we run all native.
    # Optionally add a number to offload only a specific kernel.
    args = sys.argv[1:]
    if '--hb' in args:
        on_hb = True
        args.remove('--hb')
        if args:
            # The index of the specific kernel to offload.
            kernel_idx = int(args[0])
        else:
            kernel_idx = None
    else:
        on_hb = False

    # Set up HammerBlade cosim stuff.
    if on_hb:
        torch.hammerblade.init()

        # Set up HammerBlade "routing," which tells kernels to run on HB
        # instead of on the CPU.
        with open(ROUTE_JSON) as f:
            route_data = json.load(f)
        for i, kernel in enumerate(route_data):
            # Mark kernel for offload.
            if kernel_idx is None or kernel_idx == i:
                print('offloading kernel', kernel['signature'])
                kernel['offload'] = True

            # Set up a "chart" "beacon" (?).
            torch.hammerblade.profiler.chart.add(kernel['signature'])

        torch.hammerblade.profiler.route.set_route_from_json(route_data)

    # Set the size of the run. Use TOTAL_DOCS/data_fraction of the data.

    data_fraction = HB_DATA_FRAC if on_hb else 1  # Tiny subset on HB.
    n_docs = TOTAL_DOCS // data_fraction

    # Load data and run the kernel.
    print('loading data for {} docs'.format(n_docs))
    r, cT, vecs = load_data(n_docs)
    print('done loading data; running kernel')

    start_time = begin_profile(on_hb)
    scores = swmd_torch(r, cT, vecs, niters=N_ITERS)
    end_profile(on_hb, start_time)
    if (SAVE_FILE):
        torch.save(scores, SAVE_FILE)

    # Dump profiling results, including both the overall statistics and the
    # invocation "tree" that breaks down every call stack.
    print(torch.hammerblade.profiler.stats(trimming=True))
    print(torch.hammerblade.profiler.exec_time.raw_stack())

    print("done")

# torch.hammerblade.profiler.chart.add("at::Tensor at::SparseCPUType::{anonymous}::dstmm(const at::Tensor&, const at::Tensor&)")
# torch.hammerblade.profiler.chart.add("at::Tensor at::SparseCPUType::{anonymous}::dstmmt(const at::Tensor&, const at::Tensor&)")
# torch.hammerblade.profiler.chart.add("at::Tensor at::SparseCPUType::{anonymous}::sddtmm(const at::Tensor&, const at::Tensor&, const at::Tensor&)")

# print(torch.hammerblade.profiler.exec_time.raw_stack())
# print(torch.hammerblade.profiler.chart.json())

if __name__ == '__main__':
    sinkhorn_test()
