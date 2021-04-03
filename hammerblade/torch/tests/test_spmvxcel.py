import sys
from time import time
import numpy as np
import torch 
import torch.nn.functional as F 
from time import time
import scipy
from scipy import sparse
    
def read_sparse_matrix(csr_sp):
    coo_sp = sparse.csr_matrix.tocoo(csr_sp)
    row = coo_sp.shape[0]
    col = coo_sp.shape[1]
    values = coo_sp.data
    indices = np.append([coo_sp.row], [coo_sp.col], axis=0)
    indices = torch.from_numpy(indices).long()
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).int(), torch.Size([row, col]))
    return t

def test_spmv_1():
    csr_sp = sparse.load_npz('cora_csr_float32.npz')
    coo_sp = read_sparse_matrix(csr_sp)
    coo_sp = coo_sp.coalesce()
    t = time()
    cpu_c2sr = coo_sp.to_spmvxf()
    torch.set_printoptions(profile='full')
    print(cpu_c2sr)
    convert_time = time() - t
    print('Convert time = %f' % convert_time, file=sys.stderr)
    num = cpu_c2sr.numel()
    
    col = csr_sp.shape[1]
#    cpu_v = torch.ones(col, dtype=torch.int32)

    v1 = torch.rand(col)
    v2 = torch.rand(col)
    cpu_v = (v1 + v2) * v2
    cpu_int = cpu_v.int()
    cpu_out = torch.mv(coo_sp, cpu_int)
    print(cpu_out)

    hb_c2sr = cpu_c2sr.hammerblade()
    hb_v = cpu_v.hammerblade()
#    torch.hb_trace_enable()
#    torch.hammerblade.profiler.enable()
    hb_out = torch.spmvx(hb_c2sr, hb_v)
#    torch.hammerblade.profiler.disable()
#    torch.hb_trace_disable()
    cpu_out1 = hb_out.to("cpu")
    print(cpu_out1)
    assert hb_out.device == torch.device("hammerblade")
    assert torch.allclose(cpu_out, cpu_out1)

#def test_spmv_2():
#    m = torch.nn.Threshold(0.8, 0)
#    input = torch.rand(64, 32)
#    xv = torch.rand(32)
#    xs = m(input).to_sparse()
#    print(xs)
#
#    xr = torch.mv(xs, xv)
#    print(xr)
#    hb_xv = xv.hammerblade()
#    hb_xs = xs.hammerblade()
#    hb_xr = torch.mv(hb_xs, hb_xv)
#    print(hb_xr)
#    cpu_r = hb_xr.to("cpu")

#    assert hb_xr.device == torch.device("hammerblade")
#    assert torch.allclose(cpu_r, xr)

