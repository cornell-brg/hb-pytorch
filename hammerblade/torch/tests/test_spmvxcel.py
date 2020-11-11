import numpy as np
import torch
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

def test_sptenor_blocking():
    csr_sp = sparse.load_npz('cora_csr_float32.npz')
    cpu_m = read_sparse_matrix(csr_sp)
    cpu_m = cpu_m.coalesce()
    col_np = csr_sp.shape[1]
    cpu_v = torch.ones(col_np, dtype=torch.int32)

    cpu_tensors = torch.sptensor_blocking(cpu_m, cpu_v)
    cpu_result = cpu_tensors[0]
    cpu_c2sr = cpu_tensors[1]
    cpu_vector = cpu_tensors[2]
    cpu_len_record = cpu_tensors[3]
    cpu_other_info = cpu_tensors[4]
    torch.set_printoptions(profile="full")
    print(cpu_v)
    print(cpu_vector)
#    print(cpu_c2sr)
#    print(cpu_len_record)
#    print(cpu_other_info)

    hb_result = cpu_result.hammerblade()
    hb_c2sr   = cpu_c2sr.hammerblade()
    hb_vector = cpu_vector.hammerblade()
    hb_len_record = cpu_len_record.hammerblade()
    hb_other_info = cpu_other_info.hammerblade()
    
    hb_result = torch.xcelspmv(hb_result, hb_c2sr, hb_vector, hb_len_record, hb_other_info)
   
    
