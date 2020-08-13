"""
Profiler tests for dense-sparseT matrix product (dstmm) kernel
08/13/2020 Andrew Pareles (amp342@cornell.edu)
"""

DIR = '/home/amp342/Cosim/bsg_bladerunner/hb-pytorch/hammerblade/torch/tests/sinkhorn_tests/'

import torch
import json
with open(DIR + 'cmd_sddmm.json',) as f:
    route = json.load(f)
    torch.hammerblade.profiler.route.set_route_from_json(route)

torch.manual_seed(42)

def test_torch_dstmm(a, bT):
    ah = a.hammerblade()
    bTh = bT.hammerblade()

    torch.hammerblade.profiler.enable()

    torch.dstmm(ah, bTh)

    torch.hammerblade.profiler.disable()

a = torch.Tensor([[1, 2, 3]])
b = torch.Tensor([[0, 0, 0, 4], [0, 1, 2, 0], [0, 0, 0, 1]]).t().to_sparse()
test_torch_dstmm(a, b)
