import numpy
import scipy.sparse
from scipy.spatial.distance import cdist
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from utils import parse_model_args, train, inference, save_model  # noqa
from time import time

torch.hammerblade.init()

import json
with open('sinkhorn_wmd.json',) as route:
    data = json.load(route)

# Kernel parameters.
N_FRACTION = 16*16 # use N_DOCS/N_FRACTION of the data
N_DOCS = int(4096 / N_FRACTION)
QUERY_IDX = 5
LAMBDA = 1

# Data files. (Ask Adrian for these.)
TESTS = '/home/amp342/Cosim/bsg_bladerunner/hb-pytorch/hammerblade/torch/tests/'
DATA_MAT = TESTS + 'sinkhorn_tests/sinkhorn_app/data/cache-mat.npz'
DATA_VECS = TESTS + 'sinkhorn_tests/sinkhorn_app/data/cache-vecs.npy'


def swmd_torch(r, cT, vecs, niters):
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

    # BEGIN PROFILING HERE
    start_time = time()
    torch.hammerblade.profiler.route.set_route_from_json(data)
    torch.hammerblade.profiler.enable()

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

    out = (uT.t() * torch.dstmm(K * M, vT)).sum(axis=0)
    # out = (uT * (vT @ (K_T * M.t())).sum(axis=1) 
    #Note: M is huge compared to uT, so use the sum(axis=0) instead of sum(axis=1) line
    end_time = time()
    elapsed = end_time - start_time
    print("elapsed:", elapsed)
    print("elapsed * 16:", elapsed * 16)

    return out


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

torch.hammerblade.profiler.chart.add("at::Tensor at::SparseCPUType::{anonymous}::dstmm(const at::Tensor&, const at::Tensor&)")
torch.hammerblade.profiler.chart.add("at::Tensor at::SparseCPUType::{anonymous}::dstmmt(const at::Tensor&, const at::Tensor&)")
torch.hammerblade.profiler.chart.add("at::Tensor at::SparseCPUType::{anonymous}::sddtmm(const at::Tensor&, const at::Tensor&, const at::Tensor&)")

scores = swmd_torch(r, cT, vecs, niters=1)
torch.hammerblade.profiler.disable()
# END PROFILING HERE

# print(torch.hammerblade.profiler.exec_time.raw_stack())
print(torch.hammerblade.profiler.chart.json())
print("done")
print("Multiply sddtmm, dstmmt, and dstmm times by ", N_FRACTION, " for true time on real dataset.")
