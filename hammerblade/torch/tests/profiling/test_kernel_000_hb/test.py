"""
Profiler tests for sddmm
08/12/2020 Andrew Pareles (amp342@cornell.edu)
"""

DIR = '/home/amp342/Cosim/bsg_bladerunner/hb-pytorch/hammerblade/torch/tests/sinkhorn_tests/'
import torch
# import scipy.sparse
# import numpy
import json
with open(DIR + 'cmd_sddmm.json',) as f:
    route = json.load(f)
    torch.hammerblade.profiler.route.set_route_from_json(route)

torch.manual_seed(42)

DATA_MAT = DIR + 'data/cache-mat.npz'
DATA_VECS = DIR + 'data/cache-vecs.npy'


def test_torch_sddmm(a, b, c):
    ah = a.hammerblade()
    bh = b.hammerblade()
    ch = c.hammerblade()

    torch.hammerblade.profiler.enable()

    torch.sddmm(ah, bh, ch)

    torch.hammerblade.profiler.disable()

# vecs = numpy.load(DATA_VECS)
# mat = scipy.sparse.load_npz(DATA_MAT)
# mat = mat[:, :N_DOCS]  # Use a subset of the data.

a = torch.FloatTensor([[1, 0, 1], [0, 3, 0]]).to_sparse()
b = torch.FloatTensor([[5, 3], [1, 7]])
c = torch.FloatTensor([[1, 2, 1], [2, 1, 1]])
test_torch_sddmm(a, b, c)
