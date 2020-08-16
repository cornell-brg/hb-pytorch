#!/usr/bin/env python

"""
    lgc/main.py
    
    Note to program performers:
        - parallel_pr_nibble produces the same results as ligra's `apps/localAlg/ACL-Sync-Local-Opt.C`
        - ista produces the same results as LocalGraphClustering's `ista_dinput_dense` method
"""

import os
#os.environ["OMP_NUM_THREADS"] = "1"
#os.environ["OPENBLAS_NUM_THREADS"] = "1"
#os.environ["MKL_NUM_THREADS"] = "1" # export MKL_NUM_THREADS=6
#os.environ["VECLIB_MAXIMUM_THREADS"] = "1" # export VECLIB_MAXIMUM_THREADS=4
#os.environ["NUMEXPR_NUM_THREADS"] = "1" # export NUMEXPR_NUM_THREADS=6
import sys
import argparse
import numpy as np
import torch
from time import time
from tqdm import tqdm
from scipy import sparse
from scipy.io import mmread
from scipy.stats import spearmanr
import json

# --
# Parallel PR-Nibble

def parallel_pr_nibble(seeds, adj, alpha, epsilon):
    out = []
    for seed in tqdm(seeds):
        degrees   = np.asarray(adj.sum(axis=-1)).squeeze().astype(int)
        num_nodes = adj.shape[0]
        
        p = np.zeros(num_nodes)
        r = np.zeros(num_nodes)
        r[seed] = 1
        
        frontier = np.array([seed])
        while True:
            if len(frontier) == 0:
                break
            
            r_prime = r.copy()
            for node_idx in frontier:
                p[node_idx] += (2 * alpha) / (1 + alpha) * r[node_idx]
                r_prime[node_idx] = 0
            
            for src_idx in frontier:
                neighbors = adj.indices[adj.indptr[src_idx]:adj.indptr[src_idx + 1]]
                for dst_idx in neighbors:
                    update = ((1 - alpha) / (1 + alpha)) * r[src_idx] / degrees[src_idx]
                    r_prime[dst_idx] += update
                    
            r = r_prime
            
            frontier = np.where((r >= degrees * epsilon) & (degrees > 0))[0]
        
        out.append(p)
    
    return np.column_stack(out)

# --
# ISTA algorithm

def ista(seeds, adj, alpha, rho, iters):
    out = []
    t = time()
    # Compute degree vectors/matrices
    d       = np.asarray(adj.sum(axis=-1)).squeeze()
    d_sqrt  = np.sqrt(d)
    dn_sqrt = 1 / d_sqrt

    D       = sparse.diags(d)
    Dn_sqrt = sparse.diags(dn_sqrt)
    # Normalized adjacency matrix
    Q = D - ((1 - alpha) / 2) * (D + adj)
    Q = Dn_sqrt @ Q @ Dn_sqrt
    numpy_time = time() - t
    print('pure numpy cpu time = %f' % numpy_time, file=sys.stderr)
    for seed in tqdm(seeds):
        # Make personalized distribution
        s = np.zeros(adj.shape[0])
        s[seed] = 1
        # Initialize
        q = np.zeros(adj.shape[0], dtype=np.float64)
        
        rad   = rho * alpha * d_sqrt
        grad0 = -alpha * dn_sqrt * s
        grad  = grad0
        
        # Run
        for _ in range(iters):
            q    = np.maximum(q - grad - rad, 0)
            grad = grad0 + Q @ q
        
        out.append(q * d_sqrt)
    
    return np.column_stack(out)

def spy_sparse2torch_sparse(data):
    row = data.shape[0]
    col = data.shape[1]
    values = data.data
    coo_data = data.tocoo()
    indices = np.append([coo_data.row], [coo_data.col], axis=0)
    indices = torch.from_numpy(indices).long()
#    indices = torch.LongTensor([coo_data.row, coo_data.col])
    t = torch.sparse.FloatTensor(indices, torch.from_numpy(values).float(), torch.Size([row, col]))
    return t

def test_torch_ista(seeds, adj, size, alpha, rho, iters):
    out = torch.empty(0)
    t = time()
    # Compute degree vectors/matrices
    d       = np.asarray(adj.sum(axis=-1)).squeeze()
    d_sqrt  = np.sqrt(d)
    dn_sqrt = 1 / d_sqrt

    D       = sparse.diags(d)
    Dn_sqrt = sparse.diags(dn_sqrt)

    # Normalized adjacency matrix
    Q = D - ((1 - alpha) / 2) * (D + adj)
    Q = Dn_sqrt @ Q @ Dn_sqrt
    # Convert numpy float64 data to torch float32 tensor
    t1 = time()
    Q = spy_sparse2torch_sparse(Q)
    d_sqrt = torch.from_numpy(d_sqrt).float()
    dn_sqrt = torch.from_numpy(dn_sqrt).float()
    numpy_time = t1 - t
    convert_time = time() - t1
    print('numpy cpu time = %f' % numpy_time, file=sys.stderr)
    print('numpy to pytorch convertion time = %f' % convert_time, file=sys.stderr)
    for seed in tqdm(seeds):
        s = np.zeros(adj.shape[0])
        s[seed] = 1
        s = torch.from_numpy(s).float()
        q = torch.zeros(size)        
        rad = rho * alpha * d_sqrt
        grad0 = -alpha * dn_sqrt * s
        grad = grad0.clone()
        zero = torch.zeros(size)
        for _ in range(iters):
            q = torch.max(q - grad - rad, zero)
            grad = grad0 + torch.mv(Q, q)
        temp = torch.mul(q, d_sqrt)
        temp = temp.view(size, 1)
        out = torch.cat((out, temp), 1)
        
    return out

# --
# Run

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-seeds', type=int, default=50)
    parser.add_argument('--alpha', type=float, default=0.15)
    parser.add_argument('--pnib-epsilon', type=float, default=1e-6)
    parser.add_argument('--ista-rho', type=float, default=1e-5)
    parser.add_argument('--ista-iters', type=int, default=50)
    args = parser.parse_args()
    
    # !! In order to check accuracy, you _must_ use these parameters !!
    assert args.num_seeds == 50
    assert args.alpha == 0.15
    assert args.pnib_epsilon == 1e-6
    assert args.ista_rho == 1e-5
    assert args.ista_iters == 50
    
    return args


if __name__ == "__main__":
    args = parse_args()
    
    # --
    # IO
    torch.hammerblade.init()
    array = mmread('data/jhu.mtx')
    adj = array.tocsr()
    torchadj = torch.from_numpy(array.toarray()).to_sparse()
    size = torchadj.size(0)
    
    # PNIB: Use first `num_seeds` nodes as seeds
    # ISTA: Faster algorithm, so use more seeds to get roughly comparable total runtime
#    pnib_seeds = list(range(args.num_seeds))
    ista_seeds = list(range(16))
    
    # --
    # Run Parallel PR-Nibble
    
#    t = time()
#    pnib_scores = parallel_pr_nibble(pnib_seeds, adj, alpha=args.alpha, epsilon=args.pnib_epsilon)
#    assert pnib_scores.shape[0] == adj.shape[0]
#    assert pnib_scores.shape[1] == len(pnib_seeds)
#    pnib_elapsed = time() - t
#    print('parallel_pr_nibble: elapsed = %f' % pnib_elapsed, file=sys.stderr)
    
    # --
    # Run ISTA
    
    t = time()
    ista_scores = ista(ista_seeds, adj, alpha=args.alpha, rho=args.ista_rho, iters=args.ista_iters)
    assert ista_scores.shape[0] == adj.shape[0]
    assert ista_scores.shape[1] == len(ista_seeds)
    ista_elapsed = time() - t
    print('ista: elapsed = %f' % ista_elapsed, file=sys.stderr)
    t1 = time()

#    with open('lgc_ista.json') as route:
#        data = json.load(route)
#    torch.hammerblade.profiler.route.set_route_from_json(data)
#    torch.hammerblade.profiler.enable()
    t = time()
    torch_out = test_torch_ista(ista_seeds, adj, size, alpha=args.alpha, rho=args.ista_rho, iters=args.ista_iters)
#    torch.hammerblade.profiler.disable()

    torch_ista_elapsed = time() - t1
    print('torch ista: elapsed = %f' % torch_ista_elapsed, file=sys.stderr)
    np_out = torch.from_numpy(ista_scores).float()
    assert torch.allclose(torch_out, np_out)
    
    
    # --
    # Write output
    
    os.makedirs('results', exist_ok=True)
    
    #np.savetxt('results/pnib_score.txt', pnib_scores)
    np.savetxt('results/ista_score.txt', ista_scores)
    
    #open('results/pnib_elapsed', 'w').write(str(pnib_elapsed))
    open('results/ista_elapsed', 'w').write(str(ista_elapsed))

