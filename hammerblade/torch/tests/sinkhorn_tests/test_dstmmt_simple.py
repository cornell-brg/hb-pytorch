"""
Profiler tests for dense-sparseT matrix product (dstmmt) kernel
08/18/2020 Andrew Pareles (amp342@cornell.edu)
"""

DIR = '/home/amp342/Cosim/bsg_bladerunner/hb-pytorch/hammerblade/torch/tests/sinkhorn_tests/'
import torch
import scipy.sparse
import numpy
import json
with open(DIR + 'cmd_dstmmt.json',) as f:
    route = json.load(f)
    torch.hammerblade.profiler.route.set_route_from_json(route)

torch.manual_seed(42)

DATA_MAT = DIR + 'data/cache-mat.npz'
DATA_VECS = DIR + 'data/cache-vecs.npy'

# Kernel parameters.
N_DOCS = 5000
QUERY_IDX = 100
LAMBDA = 1

# FOR TESTING
N_DATASET_TO_USE = 50

def test_torch_dstmmt(a, b, c):
    ah = a.hammerblade()
    bh = b.hammerblade()

    torch.hammerblade.profiler.enable()

    torch.dstmmt(ah, bh)

    torch.hammerblade.profiler.disable()

vecs = numpy.load(DATA_VECS)
mat = scipy.sparse.load_npz(DATA_MAT)
mat = mat[:, :N_DOCS]  # Use a subset of the data.
r = numpy.asarray(mat[:, QUERY_IDX].todense()).squeeze()
matT = mat.T 
cT = matT


# FOR TESTING:
cT = cT[:N_DATASET_TO_USE, :]


# sinkhorn_wmd sample
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
uT = 1.0 / xT

print("cT size", cT.size()) 
print("uT size", uT.size()) 
print("K_T size", K_T.size())


print("Running vT")
vT = cT * (1.0 / torch.sddtmm(cT, uT, K_T))
print("converting vT to sparse")
vT = dense_to_sparse(vT)

print("Done with vT, now to dstmmt, the main test:")
print("Multiply the total time by ", N_DOCS / N_DATASET_TO_USE)
test_torch_dstmmt(K_div_r, vT)
