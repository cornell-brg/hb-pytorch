import numpy
import scipy.sparse
from scipy.spatial.distance import cdist
import os
import sys
import torch
import argparse
from time import time

sys.path.append(os.path.join(os.path.dirname(__file__), os.pardir))

# Kernel parameters.
N_DOCS = 5000
QUERY_IDX = 100
LAMBDA = 1

# Data files. (Ask Adrian for these.)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'sinkhorn_wmd-data')
DATA_MAT = os.path.join(DATA_DIR, 'cache-mat.npz')
DATA_VECS = os.path.join(DATA_DIR, 'cache-vecs.npy')

# Kernel "routing" file.
ROUTE_JSON = os.path.join(os.path.dirname(__file__), 'sinkhorn_wmd.json')

def swmd_numpy(r, c, vecs, niters):
    # I=(r > 0)
    sel = r.squeeze() > 0

    # r=r(I)
    r = r[sel].reshape(-1, 1).astype(numpy.float64)

    # M=M(I,:)
    M = cdist(vecs[sel], vecs).astype(numpy.float64)

    # x=ones(length(r), size(c,2)) / length(r)
    a_dim = r.shape[0]
    b_nobs = c.shape[1]
    x = numpy.ones((a_dim, b_nobs)) / a_dim

    # K=exp(-lambda * M)
    K = numpy.exp(- M * LAMBDA)
    K_div_r = K / r
    K_T = K.T

    # This version uses a fixed number of iterations instead of running
    # until convergence.
    for it in range(niters):
        print('starting iteration {}'.format(it))

        u = 1.0 / x

        # Here's where a better implementation is possible by doing the
        # SDDMM thing and avoiding the dense matrix/matrix multiply. We do the
        # slow thing for now.
        K_T_times_u = K_T @ u
        one_over_K_T_times_u = 1 / (K_T_times_u)
        v = c.multiply(one_over_K_T_times_u)

        x = K_div_r @ v.tocsc()

    out = (u * ((K * M) @ v)).sum(axis=0)
    return out


def _dsmp(a, b):
    """Dense/sparse matrix product.
    """
    out = torch.zeros((a.shape[0], b.shape[1]))
    for k in range(b._nnz()):
        bi, bj = tuple(b._indices()[:, k].tolist())
        bx = b._values()[k]
        for i in range(a.shape[0]):
            out[i, bj] += a[i, bi] * bx
    return out



def _sddmm(a, b, c):
    """Only compute certain entries of b@c, based on the entries of a:
    For all i,j with a_ij!=0, compute (b@c)_ij, where `a` is sparse, `b` and `c`
    are dense, and `@` is matrix product. Returns a sparse matrix of (b@c)_ij.
    """
    outvals = torch.zeros(a._nnz())
    for k in range(a._nnz()):
        ai, aj = tuple(a._indices()[:, k].tolist())
        brow = b[ai, :]
        ccol = c[:, aj]
        outvals[k] = torch.dot(brow, ccol)
    return torch.sparse.FloatTensor(
        a._indices(),
        outvals,
        a.shape,
    )

def _sddmm_special(a, b, c, f):
    """Compute `a*f(b@c)` where `a` is sparse, `b` and `c` are dense,
    `*` is elementwise multiply, and `@` is matrix product, and `f` is a
    scalar function.

    For more on the SDDMM kernel, see:
    http://tensor-compiler.org/docs/machine_learning/
    """
    outvals = torch.zeros(a._nnz())
    for k in range(a._nnz()):
        ai, aj = tuple(a._indices()[:, k].tolist())
        brow = b[ai, :]
        ccol = c[:, aj]
        outvals[k] = a._values()[k] * f(torch.dot(brow, ccol))
    return torch.sparse.FloatTensor(
        a._indices(),
        outvals,
        a.shape,
    )


def swmd_torch(r, c, vecs, niters):

    torch.hammerblade.profiler.enable()
    start_time = time()

    # Convert arrays to PyTorch tensors.
    r = torch.FloatTensor(r)
    c_coo = c.tocoo()
    CT_coo = c.T.tocoo()
    c = torch.sparse.FloatTensor(
        torch.LongTensor(numpy.vstack((c_coo.row, c_coo.col))),
        torch.FloatTensor(c_coo.data),
        torch.Size(c_coo.shape),
    )
    
    cT = torch.sparse.FloatTensor(
        torch.LongTensor(numpy.vstack((CT_coo.row, CT_coo.col))),
        torch.FloatTensor(CT_coo.data),
        torch.Size(CT_coo.shape),
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
    b_nobs = c.shape[1]
    x = torch.ones((a_dim, b_nobs)) / a_dim
    xT = torch.ones((b_nobs, a_dim)) / a_dim

    # K=exp(-lambda * M)
    K = torch.exp(- M * LAMBDA)
    K_div_r = K / r
    K_T = K.T
    K_div_r_T = K_div_r.T

    sddmm_time = 0
    for it in range(niters):
        # print('starting iteration {}'.format(it))

        # u = 1.0 / x
        uT = 1.0 / xT

        # Compute `c * 1/(K_T @ u)` using a hand-rolled SDDMM.
        # v = c * (1.0 / _sddmm(c, K_T, u))
        # v = _sddmm_special(c,K_T, u,lambda x:1.0/x)
        sddmm_start = time()
        vT = _sddmm_special(cT,uT,K,lambda x:1.0/x)
        sddmm_time += time() - sddmm_start

        # PyTorch doesn't support dense/sparse matrix multiply (only
        # sparse/dense), so I had to write my own. :'(
        # x = _dsmp(K_div_r, v)
        xT = torch.sparse.mm(vT,K_div_r_T)

    # out = (u * _dsmp(K * M, v)).sum(axis=0)
    out = (uT.T * torch.sparse.mm(vT,(K*M).T).T).sum(axis=0)

    torch.hammerblade.profiler.disable()
    end_time = time()
    elapsed = end_time - start_time
    print("elapsed time:", elapsed)
    print("elapsed sddmm time per iteration:", sddmm_time/niters)
    return out


def add_args(parser):
    parser.add_argument('-n', '--niters', default=16, type=int,
                        help="number of iterations")
    parser.add_argument('-d', '--dump', default=False, action='store_true',
                        help="dump result to a file")
    parser.add_argument('-p', '--numpy', default=False, action='store_true',
                        help="use NumPy version instead of PyTorch")
    parser.add_argument('-c', '--compare', default=False, action='store_true',
                        help="compare NumPy and PyTorch output")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    add_args(parser)
    args = parser.parse_args()

    # Load data.
    vecs = numpy.load(DATA_VECS)
    mat = scipy.sparse.load_npz(DATA_MAT)
    mat = mat[:, :N_DOCS]  # Use a subset of the data.
    print('data loaded')

    # The query vector.
    r = numpy.asarray(mat[:, QUERY_IDX].todense()).squeeze()

    # The kernel itself.
    if args.compare:
        # Compare the output of the reference Numpy version and our PyTorch
        # version to ensure the output matches.
        print('starting numpy')
        scores_numpy = torch.FloatTensor(
            swmd_numpy(r, mat, vecs, niters=args.niters)
        )
        print('starting torch')
        scores_torch = swmd_torch(r, mat, vecs, niters=args.niters)
        print('done')
        print(scores_torch)
        print(scores_numpy)
        if torch.allclose(scores_torch, scores_numpy, atol=0.01):
            print('success! :)')
        else:
            print('failure :(')
            sys.exit(1)
    else:
        # Run a single version of the kernel.
        kernel = swmd_numpy if args.numpy else swmd_torch
        scores = kernel(r, mat, vecs,
                        niters=args.niters)

    # print(torch.hammerblade.profiler.stats(trimming=True))
    # Dump output.
    if args.dump:
        numpy.savetxt('scores_out.txt', scores, fmt='%.8e')
