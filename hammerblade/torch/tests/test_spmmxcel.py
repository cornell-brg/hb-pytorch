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
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), torch.Size([row, col]))
    return t

def test_spmmxcel():
    csr_sp = sparse.load_npz('sinkhorn_csr_float32.npz')
    coo_sp = read_sparse_matrix(csr_sp)
    coo_sp = coo_sp.coalesce()
    t = time()
    cpu_c2sr = coo_sp.to_spmmxf()
    torch.set_printoptions(profile='full')
    print(cpu_c2sr)
    convert_time = time() - t
    print('Convert time = %f' % convert_time, file=sys.stderr)
    num = cpu_c2sr.numel()
    
#    col = csr_sp.shape[1]

#    v1 = torch.rand(col)
#    v2 = torch.rand(col)
#    cpu_v = (v1 + v2) * v2
#    cpu_int = cpu_v.int()
    cpu_float = torch.randn(100000, 19) * 10
    cpu_int = cpu_float.int()
#    cpu_out = torch.mm(coo_sp, cpu_int)
#    print(cpu_out)

    hb_c2sr = cpu_c2sr.hammerblade()
    hb_m = cpu_int.hammerblade()
#    torch.hb_trace_enable()
#    torch.hammerblade.profiler.enable()
    hb_out = torch.spmmx(hb_c2sr, hb_m)
    print(hb_out.size())
#    torch.hammerblade.profiler.disable()
#    torch.hb_trace_disable()
    cpu_out1 = hb_out.to("cpu")
#    print(cpu_out1)
#    assert hb_out.device == torch.device("hammerblade")
#    assert torch.allclose(cpu_out, cpu_out1)

#def test_spmm_manycore():
#    csr_sp = sparse.load_npz('sinkhorn_csr_float32.npz')
#    coo_sp = read_sparse_matrix(csr_sp)
#    a = coo_sp.coalesce()
#    nnz = a._nnz()
#    print(nnz)
#    b = torch.rand(100000, 19)
#    r = torch.mm(coo_sp, b)

#    hb_a = a.hammerblade()
#    hb_b = b.hammerblade()
#    hb_r = torch.mm(hb_a, hb_b)

#    assert hb_r.device == torch.device("hammerblade")
#    assert torch.allclose(r, hb_r.cpu())

