import numpy
import scipy.sparse
import os
import sys
import torch
import json

from test_sinkhorn import load_data

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--cosim_scale",type=int,default=1,help="reduces kernel compute by this factor. Need to scale timings later for 1 pod")
args = parser.parse_args()

# Data files. (Ask Adrian for these.)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'sinkhorn_wmd-data')
DATA_MAT = os.path.join(DATA_DIR, 'cache-mat.npz')
DATA_VECS = os.path.join(DATA_DIR, 'cache-vecs.npy')

TOTAL_DOCS = 5000
HB_DATA_FRAC = 16
QUERY_IDX = 100
COSIM_SCALE = args.cosim_scale #10

n_docs = TOTAL_DOCS
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

x1 = vecs[sel].hammerblade()
x2 = vecs[:(vecs_row//HB_DATA_FRAC)//COSIM_SCALE,:].hammerblade()  #hammerblade weak scaling
# print(x1.size(),x2.size())
cdist_h = torch.cdist(x1,x2)

# assert cdist_h.device == torch.device("hammerblade") #might be bad for profiling data!
torch.hammerblade.profiler.disable()

print(torch.hammerblade.profiler.stats(trimming=True))
print(torch.hammerblade.profiler.exec_time.raw_stack())


