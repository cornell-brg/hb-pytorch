import numpy
import scipy.sparse
from scipy.spatial.distance import cdist
import os
import sys
import torch

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))
from utils import parse_model_args, train, inference, save_model  # noqa

# Kernel parameters.
N_DOCS = 5000
QUERY_IDX = 100
LAMBDA = 1

# Data files. (Ask Adrian for these.)
DATA_MAT = '/home/amp342/Cosim/hb-pytorch/hammerblade/torch/tests/sinkhorn_tests/sinkhorn_app/data/cache-mat.npz'
DATA_VECS = '/home/amp342/Cosim/hb-pytorch/hammerblade/torch/tests/sinkhorn_tests/sinkhorn_app/data/cache-vecs.npy'

def dense_to_sparse(m):
    m_coo = m.tocoo()
    return torch.sparse.FloatTensor(
        torch.LongTensor(numpy.vstack((m_coo.row, m_coo.col))),
        torch.FloatTensor(m_coo.data),
        torch.Size(m_coo.shape),
    )


def swmd_torch(r, cT, vecs, niters):
    # Convert arrays to PyTorch tensors.
    r = torch.FloatTensor(r)
    c_coo = cT.tocoo()
    cT = torch.sparse.FloatTensor(
        torch.LongTensor(numpy.vstack((c_coo.row, c_coo.col))),
        torch.FloatTensor(c_coo.data),
        torch.Size(c_coo.shape),
    )

    vecs = torch.FloatTensor(vecs)

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
        vT = cT * (1.0 / torch.sddtmm(cT, uT, K_T)

        # in the future, vT should return a sparse tensor. Since that's not supported, for now, we convert it to one
        vT = dense_to_sparse(vT)

        # custom dstmm.t():
        # x = _dsmp(K_div_r, v)
        # x = torch.dstmm(K_div_r, vT)
        xT = torch.dstmmt(K_div_r, vT)
        
    out = (uT.t() * dstmm(K * M, vT)).sum(axis=0)
    # out = (u * (v @ (K_T * M_T)).sum(axis=1)
    return out



# Load data.
vecs = numpy.load(DATA_VECS)
mat = scipy.sparse.load_npz(DATA_MAT)
mat = mat[:, :N_DOCS]  # Use a subset of the data.

# The query vector.
r = numpy.asarray(mat[:, QUERY_IDX].todense()).squeeze()

# mat could theoretically be stored as its transpose, so don't count 
matT = mat.T 
# BEGIN PROFILING HERE
scores = swmd_torch(r, matT, vecs, niters=1)
# END PROFILING HERE
