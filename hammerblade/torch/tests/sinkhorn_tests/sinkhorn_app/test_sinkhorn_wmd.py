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


def swmd_torch(r, c, vecs, niters):
    # Convert arrays to PyTorch tensors.
    r = torch.FloatTensor(r)
    c_coo = c.tocoo()
    c = torch.sparse.FloatTensor(
        torch.LongTensor(numpy.vstack((c_coo.row, c_coo.col))),
        torch.FloatTensor(c_coo.data),
        torch.Size(c_coo.shape),
    )
    cT = c.t()

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
    x = torch.ones((a_dim, b_nobs)) / a_dim

    # K=exp(-lambda * M)
    K = torch.exp(- M * LAMBDA)
    K_div_r = K / r

    for it in range(niters):
        print('starting iteration {}'.format(it))

        u = 1.0 / x

        # Interesting property: sddtmm(a,b,c).T = sddtmm(a.T,c,b)
        # Compute `c * 1/(K_T @ u)` using a hand-rolled SDDMM.
        # v = c * (1.0 / _sddmm(c, K_T, u))
        # v = c * (1.0 / torch.sddtmm(c, K_T, u.t()))
        # vT = cT * (1.0 / torch.sddtmm(cT, uT, K_T))
        # vT = cT * (1.0 / torch.stddtmm(c, uT, K_T))
        vT = cT * (1.0 / torch.stddtmmt(cT, K, u))
        # in the future, vT should return a sparse tensor. Since that's not supported, for now, we convert it to one
        vT = dense_to_sparse(vT)
        # custom dstmm.t():
        # x = _dsmp(K_div_r, v)
        # x = torch.dstmm(K_div_r, vT)
        x = torch.dstmmt(K_div_r, vT)

    out = (u * dstmm(K * M, vT)).sum(axis=0)
    return out



# Load data.
vecs = numpy.load(DATA_VECS)
mat = scipy.sparse.load_npz(DATA_MAT)
mat = mat[:, :N_DOCS]  # Use a subset of the data.

# The query vector.
r = numpy.asarray(mat[:, QUERY_IDX].todense()).squeeze()


# BEGIN PROFILING HERE
scores = swmd_torch(r, mat, vecs,niters=1)
# END PROFILING HERE