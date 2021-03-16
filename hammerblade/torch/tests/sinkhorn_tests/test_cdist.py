import numpy
import scipy.sparse
import os
import sys
import torch
import json

from test_sinkhorn import load_data

# Data files. (Ask Adrian for these.)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'sinkhorn_wmd-data')
DATA_MAT = os.path.join(DATA_DIR, 'cache-mat.npz')
DATA_VECS = os.path.join(DATA_DIR, 'cache-vecs.npy')

TOTAL_DOCS = 5000
HB_DATA_FRAC = 16
QUERY_IDX = 100

n_docs = TOTAL_DOCS // HB_DATA_FRAC
# Load data and run the kernel.
print('loading data for {} docs'.format(n_docs))
r, cT, vecs = load_data(n_docs)
print('done loading data; running cdist kernel')

#setting up the input data to cdist
sel = r > 0
r = r[sel].reshape(-1, 1)
vecs_row, _ = vecs.shape

torch.hammerblade.profiler.enable()
# #sinkhorn matrix values, x2.size(0) is 16x lesser than orig for weak pod scaling
# x1 = torch.randn(19,300).hammerblade()
# x2 = torch.randn((100000//16),300).hammerblade()

x1 = vecs[sel].hammerblade()
x2 = vecs[:vecs_row//16,:].hammerblade()  #hammerblade weak scaling

cdist_h = torch.cdist(x1,x2)

# assert cdist_h.device == torch.device("hammerblade") #might be bad for profiling data!
torch.hammerblade.profiler.disable()

print(torch.hammerblade.profiler.stats(trimming=True))
print(torch.hammerblade.profiler.exec_time.raw_stack())


