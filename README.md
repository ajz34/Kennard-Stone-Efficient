# Efficient Kennard-Stone Sampling Algorithms

## Introduction

Kennard-Stone (KS) sampling is usually utilized in training/validation dataset splitting.
It is generally considered more powerful than random dataset splitting, and can improve
generalization error when training.

However, this algorithms is mostly used in quite small or moderate datasets (<5000 samples).
These days, even larger datasets becomes common. This algorithms is seldom used with these
datasets.

This repository is aimed to resolve KS sampling with quite large datasets.
(~30k in 6 seconds if memory is abundant for personal computer, or ~131k in one hour if memory is limited)

Kennard-Stone sampling algorithms original article:
> Kennard, R. W.; Stone, L. A. Computer Aided Design of Experiments, **1969**, *11*(1), 137-148. doi: 10.1080/00401706.1969.10490666

Implementation and demonstration details (in Chinese blog):
> https://ajz34.readthedocs.io/zh_CN/latest/ML_Notes/Kennard-Stone/Kennard-Stone-ON2.html
> https://ajz34.readthedocs.io/zh_CN/latest/ML_Notes/Kennard-Stone/Kennard-Stone-Mem.html


## Implementation, Capacity and Highlights

There are two implementations in this repository.

**Fast Scheme** requires full distance matrix stored in memory (n_sample * n_sample).

The capacity and highlights of this implementation are:

- Algorithms complexity is O(N^2), or O(n_sample n_result), where n_sample refers to number
  of samples, and n_result refers to number of results.
- Ultimately fast. For a **30k dataset**, Kennard-Stone sampling (without distance matrix
  formation) finishes well quicker than 20 seconds. For my personal computer (Intel-i5
  7300 HQ, python code ~15s, C code with 4-core parallel ~2s; where distance matrix needs ~4s). 
  - This could mean with enough memory capacity, time cost of KS sampling for even larger
    dataset is totally affordable.
  - The reason of using 30k dataset is that it's near my computer's memory capacity.
- Euclidean-metric (L2 norm) distance matrix could be faster than implementation of
  `sklearn.metrics.pairwise`. Using default threading parallelization of numpy.
- Kennard-Stone sampling utilizes C code and OpenMP threading parallelization.

**Memory Limited Scheme** requires full data stored in memory (n_sample * n_feature)
and additional intermediate memory (should be no more than (n_core + 10) * n_sample).

The capacity and highlights of this implementation are:

- Algorithms complexity is O(T N^2), or O(n_feature n_sample n_result), where n_feature refers
  to lenght of feature vector. No need of full distance matrix.
- Acceptable speed. For a **131k dataset** (QM9), and Coulumb Matrix feature (n_feature=900),
  my computer finishes KS sampling in one hour.
- Fully parallelized to find the two maximum distant samples using multiprocessing, avoiding
  any parallelization deficiency of numpy distance-matrix generation.
- Kennard-Stone sampling utilizes C code and OpenMP threading parallelization.
- **Only Euclidean-metric is implemented for memory limited scheme.**


## Usage

**Preparation**

`KS_Sampling` uses ctype to achieve C-binding. Use this function if you can use `gcc` compiler.
Higher version of `gcc` is prefered, since C program uses OpenMP/reduce.

`KS_Sampling_pure_python` only uses Python language. That can be a lot slower, but can be used
in any condition.

Python package `numpy` is required. `pathos` is required in memory limited scheme.
If memory is not your problem, you may not need memory limited scheme, and remoe `pathos` dependency.

**Fast Scheme**

Suppose `X` is original data, with 1000 samples, each length of feature vector is 20.
Now we want to select 800 samples (80% split), then execuate

```python
from KS_Sampling import ks_sampling
# X.shape = (1000, 20)
ks_result = ks_sampling(X, n_result=800)
```

`ks_result[0]` contains selected samples by generation sequence.
`ks_result[1]` contains minimum distance of selected points when generated.

**Memory Limited Scheme**

```python
from KS_Sampling import ks_sampling_mem
# X.shape = (1000, 20)
ks_result = ks_sampling_mem(X, n_result=800)
```
