import numpy as np
import numpy.ma as ma
import ctypes
import os
import os.path as path
from pathos.multiprocessing import ProcessingPool as Pool
from sklearn.metrics import pairwise_distances


prog_dir = path.dirname(path.abspath(__file__))
cpp_name = "ks_cpp"

if not os.path.isfile(prog_dir + "/" + cpp_name + ".so"):
    current_dir = os.path.abspath(".")
    os.chdir(path.dirname(path.abspath(__file__)))
    os.system("gcc -fPIC  -fopenmp -O3 -shared -o" + cpp_name + ".so " + cpp_name + ".c")
    os.chdir(current_dir)
ks_cpp = np.ctypeslib.load_library(cpp_name + ".so", prog_dir)


def get_dist_unsafe(X):
    """
    This implementation of distance matrix is fast but unsafe.
    Numerical discrepency could occur rarely.
    """
    dist = X @ X.T
    t = dist.diagonal().copy()
    dist *= -2
    dist += t[:, None]
    dist += t[None, :]
    return np.sqrt(dist)


def ks_sampling(X, seed=None, n_result=None, get_dist=pairwise_distances, backend="C"):
    """
    ks_sampling_general(X, seed=None, n_result=None, backend="C")

    Kennard-Stone Full Sampling Program

    Parameters
    ----------

    X: np.ndarray, shape: (n_sample, n_feature)
        Original data, need to be generated by user.

    seed: np.ndarray or list or None, shape: (n_seed, ), optional
        Initial selected seed.
        If set as `None`, the program will find the two samples
        which have largest distance as seed.

    n_result: int or None, optional
        Number of samples that should be selected.
        If set as `None`, `n_sample` will be used instead, i.e.
        selectet all data.

    get_dist: function
        A function `get_dist(X)` that will read original data, and
        return distance.
        Default Implementation is scikit-learn Euclidean distance.

    backend: str, "Python" or "C"
        Specify Kennard-Stone sampling function backend in Python
        language or C language.
    """
    X = np.asarray(X, dtype=float)
    if n_result is None:
        n_result = X.shape[0]
    dist = get_dist(X)
    if backend == "Python":
        if seed is None or len(seed) == 0:
            seed = np.unravel_index(np.argmax(dist), dist.shape)
        return ks_sampling_core(dist, seed, n_result)
    elif backend == "C":
        return ks_sampling_core_cpp(dist, seed, n_result)
    else:
        raise NotImplemented("Other backends are not implemented!")


def ks_sampling_mem(X, seed=None, n_result=None, backend="C", n_proc=4, n_batch=1000):
    """
    ks_sampling_mem(X, seed=None, n_result=None, backend="C", n_proc=4, n_batch=1000)

    Kennard-Stone Full Sampling Program
        (with limited memory)

    If user have enough memory space, using `ks_sampling`
    instead of `ks_sampling_mem` is strongly recommended.
    
    This program could possibly handle very large dataset.
    To make memory cost as low as possible, `n_batch` could
    be set to about sqrt(X.shape[0]) manually.
    However, to make efficiency as the first priority,
    `n_batch` could be set to as large as possible.
    
    NOTE! Only Euclid distance is available currently!

    Parameters
    ----------

    X: np.ndarray, shape: (n_sample, n_feature)
        Original data, need to be generated by user.

    seed: np.ndarray or list or None, shape: (n_seed, ), optional
        Initial selected seed.
        If set as `None`, the program will find the two samples
        which have largest distance as seed.

    n_result: int or None, optional
        Number of samples that should be selected.
        If set as `None`, `n_sample` will be used instead, i.e.
        selectet all data.

    backend: str, "Python" or "C"
        Specify Kennard-Stone sampling function backend in Python
        language or C language.
    
    n_proc: int, optional
        Number of Python's multiprocessing processors.
        NOTE! This variable only controls Python's code.
        User need to use OMP_NUM_THREADS in environment to specify
        C program's paralleling behavior.
    
    n_batch: int, optional
        The dimension of distance matrix evaluation in one processor.
    """
    X = np.asarray(X, dtype=float)
    n_sample = X.shape[0]
    if n_result is None:
        n_result = X.shape[0]
    # Find most distant sample indexes if no seed provided
    if seed is None or len(seed) == 0:
        t = np.einsum("ia, ia -> i", X, X)
        
        def get_dist_slice(sliceA, sliceB):
            distAB = t[sliceA, None] - 2 * X[sliceA] @ X[sliceB].T + t[None, sliceB]
            if sliceA == sliceB:
                np.fill_diagonal(distAB, 0)
            return np.sqrt(distAB)
        
        def get_maxloc_slice(slice_pair):
            dist_slice = get_dist_slice(slice_pair[0], slice_pair[1])
            max_indexes = np.unravel_index(np.argmax(dist_slice), dist_slice.shape)
            return (dist_slice[max_indexes], max_indexes[0] + slice_pair[0].start, max_indexes[1] + slice_pair[1].start)
        
        p = list(np.arange(0, n_sample, n_batch)) + [n_sample]
        slices = [slice(p[i], p[i+1]) for i in range(len(p) - 1)]
        slice_pairs = [(slices[i], slices[j]) for i in range(len(slices)) for j in range(len(slices)) if i <= j]
        
        with Pool(n_proc) as p:
            maxloc_slice_list = p.map(get_maxloc_slice, slice_pairs)
        max_indexes = maxloc_slice_list[np.argmax([v[0] for v in maxloc_slice_list])][1:]
        seed = max_indexes
    seed = np.asarray(seed, dtype=np.uintp)
    
    if backend == "Python":
        return ks_sampling_mem_core(X, seed, n_result)
    elif backend == "C":
        return ks_sampling_mem_core_cpp(X, seed, n_result)
    else:
        raise NotImplemented("Other backends are not implemented!")


def ks_sampling_core(dist, seed, n_result):
    assert(dist.shape[0] == dist.shape[1])
    n_sample = dist.shape[0]
    # Definition: Output Variables
    result = np.zeros(n_result, dtype=int)
    v_dist = np.zeros(n_result, dtype=float)
    # Definition: Intermediate Variables
    n_seed = len(seed)
    selected = np.zeros(n_sample, dtype=bool)
    min_vals = np.zeros(n_sample, dtype=float)
    # --- Initialization ---
    result[:n_seed] = seed                   # - 1
    for i in seed:
        selected[i] = True
    if n_seed == 2:
        v_dist[0] = dist[seed[0], seed[1]]   # - 2
    min_vals[:] = dist[seed[0]]              # - 3
    upper_bound = min_vals.max()             # - 4
    for n in seed:                           # - 5
        np.min(np.array([min_vals, dist[n]]), axis=0, initial=upper_bound, where=np.logical_not(selected), out=min_vals)
    # --- Loop argmax minimum ---
    for n in range(n_seed, n_result):
        sup_index = ma.array(min_vals, mask=selected).argmax()  # - 1
        result[n] = sup_index                                   # - 2
        v_dist[n - 1] = min_vals[sup_index]                     # - 3
        selected[sup_index] = True                              # - 4     # | 5
        np.min(np.array([min_vals, dist[sup_index]]), axis=0, initial=upper_bound, where=np.logical_not(selected), out=min_vals)
    return result, v_dist


def ks_sampling_core_cpp(dist, seed=None, n_result=None):
    """
    ks_sampling_core_cpp(dist, seed=None, n_result=None)
    
    Kennard-Stone Sampling Program
    
    Parameters
    ----------
    
    dist: np.ndarray
        shape: (n_sample, n_sample)
        Distances of samples, need to be generated by user.
        
    seed: np.ndarray or list or None, optional
        shape: (n_seed, )
        Initial selected seed.
        If set as `None`, the C program will find the two samples
        which have largest distance as seed.
        
    n_result: int or None, optional
        Number of samples that should be selected.
        If set as `None`, `n_sample` will be used instead.
    """
    assert(dist.shape[0] == dist.shape[1])
    n_sample = dist.shape[0]
    if n_result is None:
        n_result = n_sample
    if seed is None:
        seed = np.zeros(2, dtype=np.uintp)
        n_seed = 0
    else:
        seed = np.asarray(seed, dtype=np.uintp)
        n_seed = seed.shape[0]
    vdist = np.zeros(n_result, dtype=float)
    result = np.zeros(n_result, dtype=np.uintp)
    ks_cpp.kennard_stone(
        dist.astype(float).ctypes.data_as(ctypes.c_void_p),
        seed.ctypes.data_as(ctypes.c_void_p),
        result.ctypes.data_as(ctypes.c_void_p),
        vdist.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(n_sample),
        ctypes.c_size_t(n_seed),
        ctypes.c_size_t(n_result),
    )
    return result.astype(int), vdist


def ks_sampling_mem_core(X, seed, n_result):
    # Definition: Output Variables
    result = np.zeros(n_result, dtype=int)
    v_dist = np.zeros(n_result, dtype=float)
    
    # Definition: Intermediate Variables
    n_seed = len(seed)
    n_sample = X.shape[0]
    
    # --- Initialization ---
    def sliced_dist(idx):
        tmp_X = X[remains] - X[idx]
        return np.sqrt(np.einsum("ia, ia -> i", tmp_X, tmp_X))

    remains = []
    for i in range(n_sample):
        if i not in seed:
            remains.append(i)
    result[:n_seed] = seed
    if n_seed == 2:
        v_dist[0] = np.linalg.norm(X[seed[0]] - X[seed[1]])
    min_vals = sliced_dist(seed[0])
    
    for n in seed:
        np.min(np.array([min_vals, sliced_dist(n)]), axis=0, out=min_vals)
    # --- Loop argmax minimum ---
    for n in range(n_seed, n_result):
        sup_index = min_vals.argmax()
        result[n] = remains[sup_index]
        v_dist[n - 1] = min_vals[sup_index]
        remains.pop(sup_index)
        min_vals[sup_index:-1] = min_vals[sup_index+1:]
        min_vals = min_vals[:-1]
        np.min(np.array([min_vals, sliced_dist(result[n])]), axis=0, out=min_vals)
    return result, v_dist


def ks_sampling_mem_core_cpp(X, seed, n_result=None):
    """
    ks_sampling_mem_core_cpp(X, seed=None, n_result=None)
    
    Kennard-Stone Sampling Program
        (with limited memory, no need of distance matrix)
    
    Parameters
    ----------
    
    X: np.ndarray, shape: (n_sample, n_feature)
        Original data, need to be provided by user.
        
    seed: np.ndarray or list or None, shape: (n_seed, )
        **THIS IS NOT OPTIONAL**
        Initial selected seed.
        If set as `None`, the C program will find the two samples
        which have largest distance as seed.
        
    n_result: int or None, optional
        Number of samples that should be selected.
        If set as `None`, `n_sample` will be used instead.
    """
    n_sample = X.shape[0]
    n_feature = X.shape[1]
    if n_result is None:
        n_result = n_sample
    seed = np.asarray(seed, dtype=np.uintp)
    n_seed = seed.shape[0]
    vdist = np.zeros(n_result, dtype=float)
    result = np.zeros(n_result, dtype=np.uintp)
    ks_cpp.kennard_stone_mem(
        X.astype(float).ctypes.data_as(ctypes.c_void_p),
        seed.ctypes.data_as(ctypes.c_void_p),
        result.ctypes.data_as(ctypes.c_void_p),
        vdist.ctypes.data_as(ctypes.c_void_p),
        ctypes.c_size_t(n_sample),
        ctypes.c_size_t(n_feature),
        ctypes.c_size_t(n_seed),
        ctypes.c_size_t(n_result),
    )
    return result.astype(int), vdist
