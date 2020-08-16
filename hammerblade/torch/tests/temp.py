for seed in tqdm(seeds):
        s = np.zeros(adj.shape[0])
        s[seed] = 1

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
        Q = spy_sparse2torch_sparse(Q)
        d_sqrt = torch.from_numpy(d_sqrt).float()
        dn_sqrt = torch.from_numpy(dn_sqrt).float()
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
