"""
Computes the maximum mean discrepancy (MMD) between compressed/latent representations of two datasets.
The datasets must be stored as tensors in .npy format. 
The tensors must be at least two-dimensional, with the batch dimension assumed to be the last dimension.
"""

# Standard library
import argparse
import os
from pathlib import Path

# Other deps
import numpy as np
import scipy

# Argument parsing setup
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--datasets", nargs=2, help="Datasets in npy format between which to compute MMD")
parser.add_argument("-n", "--num-samples", type=int, help="Number of samples of datasets used to compute MMD")
parser.add_argument("-b", "--bandwidth", type=float, default=10, help="Width of radial basis function kernels used for computing MMD")
parser.add_argument("-t", "--trials", type=int, default=1, help="Number of times to estimate MMD")
parser.add_argument("-o", "--output-file", type=Path, default=Path("mmd.npy"), help="Output file into which MMD data is put")

def kernel_sum(X, Y, sigma = 10.0, zero_diag=False):
    """
    For MMD, compute distance between all pairs of points in X and Y, evaluate the squared exponential kernel, then sum.
    # Parameters
    - X, Y: Arrays of dimension (nx, mx) and (ny, my)
    - sigma: Kernel bandwidth
    """
    nx, mx = X.shape
    ny, my = Y.shape

    r2 = scipy.spatial.distance.cdist(X, Y, 'sqeuclidean')

    assert r2.shape == (nx, ny) 

    k = np.exp(-r2 / (2 * sigma**2))
    if zero_diag:
        np.fill_diagonal(k, 0.0)

    return np.sum(k)

def calc_mmd(X, Y, bandwidth = 10.0):
    (m, _) = X.shape
    (n, _) = Y.shape

    k_xx = kernel_sum(X, X, bandwidth, zero_diag=True) / (m * (m-1))
    k_yy = kernel_sum(Y, Y, bandwidth, zero_diag=True) / (n * (n-1))
    k_xy = kernel_sum(X, Y, bandwidth, zero_diag=False) / (m * n)

    return k_xx + k_yy - 2 * k_xy

def main(args):
    mmds = np.zeros(args.trials)
    X = np.load(args.datasets[0])
    Y = np.load(args.datasets[1])
    ix = np.arange(X.shape[-1])
    iy = np.arange(Y.shape[-1])

    for t in range(args.trials):
        inds_x = np.random.choice(ix, args.num_samples)
        inds_y = np.random.choice(iy, args.num_samples)        
        mmds[t] = calc_mmd(X[:, inds_x], Y[:, inds_y], bandwidth=args.bandwidth)
    np.save(args.output_file, mmds)
    
    return mmds

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)