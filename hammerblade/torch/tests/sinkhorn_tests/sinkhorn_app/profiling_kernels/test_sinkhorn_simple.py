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
        # vT = cT * torch.sddtmm(cT, uT, K_T).sparse_reciprocal()
        
        # NOTE: NEED TO ADD RECIPROCAL
        vT = cT * torch.sddtmm(cT, uT, K_T)
        
        # custom dstmm.t():
        # x = _dsmp(K_div_r, v)
        # x = torch.dstmm(K_div_r, vT)
        xT = torch.dstmmt(K_div_r, vT)

    #Note: M is huge compared to uT, so use the sum(axis=0) instead of sum(axis=1) line
    # out = (uT * (vT @ (K_T * M.t())).sum(axis=1) 
    out = (uT.t() * torch.dstmm(K * M, vT)).sum(axis=0)
    return out


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
    r, cT, vecs = load_data()
    print('done loading data; running kernel')

    begin_profile(on_hb)
    scores = swmd_torch(r, cT, vecs, niters=1)
    end_profile(on_hb, N_FRACTION)

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