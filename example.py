import os

os.system("rm ks_cpp.so")

import numpy as np
from KS_Sampling import ks_sampling, ks_sampling_mem

np.set_printoptions(precision=6, linewidth=120, suppress=True)
np.random.seed(0)


if __name__ == '__main__':
    # -- Example 1 -- 5000 data points, feature vector length 100
    n_sample = 5000
    n_feature = 100
    X = np.random.randn(n_sample, n_feature)
    X *= 100
    print(ks_sampling(X, seed=[345, 456], n_result=4990))
    print(ks_sampling(X, seed=[345, 456], n_result=4990, backend="Python"))
    print(ks_sampling_mem(X, seed=[345, 456], n_result=4990))
    print(ks_sampling_mem(X, seed=[345, 456], n_result=4990, backend="Python"))
    # (array([ 345,  456,  450, ..., 1696, 4495, 4400]),
    #  array([1388.464734, 1649.251576, 1633.396292, ...,  959.175021,  956.828118,    0.      ]))
    # -- Example 2 -- data with sets of same values
    X = np.array([[1], [1], [2], [2], [2], [3], [3]])
    print(ks_sampling(X))
    print(ks_sampling(X, backend="Python"))
    print(ks_sampling_mem(X))
    print(ks_sampling_mem(X, backend="Python"))


