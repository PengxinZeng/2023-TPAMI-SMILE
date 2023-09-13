import time

import numpy as np
import scipy
import torch
from sklearn.decomposition import PCA


def get_nearest_k(h0, h1, k=1, sp_size=1000):
    hh0 = h0.half()
    hh1 = h1.half()
    split = int(np.ceil(len(hh0) / sp_size))
    near = []
    for i in range(split):
        dist = torch.cdist(hh0[i * sp_size:(i + 1) * sp_size], hh1)
        nearest = torch.argsort(dist, dim=1)[:, :k]
        near.append(nearest)
    nearest = torch.cat(near)
    return nearest


def re_align(h0, h1, reduce_dim=0):
    if reduce_dim:
        h0_r, h1_r = [PCA(reduce_dim).fit_transform(it) for it in [h0, h1]]
    else:
        h0_r, h1_r = [h0, h1]
    sim_matrix = torch.nn.Softmin(dim=1)(torch.cdist(
        torch.from_numpy(h0_r).cuda().half(),
        torch.from_numpy(h1_r).cuda().half()
    )).cpu().numpy()
    res = scipy.optimize.linear_sum_assignment(-sim_matrix)[1]
    h1_r2 = h1[res]
    # sim_matrix = torch.nn.Softmin(dim=1)(torch.cdist(h0.cuda(), h1.cuda()))
    return h1_r2


def main():
    code_test = False
    if code_test:
        sz = [5, 1]
        n_tim = 1
    else:
        sz = [30000, 10]
        n_tim = 1

    for func in [re_align]:
        h0 = torch.randn(sz, )
        h1 = torch.randn(sz, )
        t_s = time.time()
        for _ in range(n_tim):
            func(h0, h1, reduce_dim=10)
        print((time.time() - t_s) / n_tim)


if __name__ == '__main__':
    main()
