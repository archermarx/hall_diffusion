"""
Computes the maximum mean discrepancy (MMD) between compressed/latent representations of two datasets.
The datasets must be stored as tensors in .npy format.
The tensors must be at least two-dimensional, with the batch dimension assumed to be the last dimension.
"""

# Standard library
import argparse
from pathlib import Path

# Other deps
import numpy as np
import scipy
from joblib import Parallel, delayed

# Argument parsing setup
parser = argparse.ArgumentParser()
parser.add_argument("-d", "--datasets", nargs=2, help="Datasets in npy format between which to compute MMD")
parser.add_argument("-n", "--num-samples", type=int, help="Number of samples of datasets used to compute MMD")
parser.add_argument(
    "-b", "--bandwidth", type=float, default=10, help="Width of radial basis function kernels used for computing MMD"
)
parser.add_argument("-t", "--trials", type=int, default=1, help="Number of times to estimate MMD")
parser.add_argument(
    "-o", "--output-file", type=Path, default=Path("mmd.npy"), help="Output file into which MMD data is put"
)


def calc_mmd(X, Y, bandwidth=10.0):
    m, n = len(X), len(Y)
    Z = np.vstack([X, Y])
    D = scipy.spatial.distance.pdist(Z, "sqeuclidean")
    K = np.exp(-scipy.spatial.distance.squareform(D) / (2 * bandwidth**2))

    k_xx = (K[:m, :m].sum() - m) / (m * (m - 1))  # subtract diagonal (all 1s)
    k_yy = (K[m:, m:].sum() - n) / (n * (n - 1))
    k_xy = K[:m, m:].sum() / (m * n)

    return k_xx + k_yy - 2 * k_xy


def main(args):
    X = np.load(args.datasets[0])
    Y = np.load(args.datasets[1])
    ix = np.arange(X.shape[-1])
    iy = np.arange(Y.shape[-1])

    X = X.astype(np.float32)
    Y = Y.astype(np.float32)

    # mmds = np.zeros(args.trials)
    # for t in range(args.trials):
    #     inds_x = np.random.choice(ix, args.num_samples)
    #     inds_y = np.random.choice(iy, args.num_samples)
    #     mmds[t] = calc_mmd(X[:, inds_x], Y[:, inds_y], bandwidth=args.bandwidth)

    mmds = Parallel(n_jobs=-1)(
        delayed(calc_mmd)(
            X[:, np.random.choice(ix, args.num_samples)],
            Y[:, np.random.choice(iy, args.num_samples)],
            bandwidth=args.bandwidth,
        )
        for _ in range(args.trials)
    )
    np.save(args.output_file, mmds)

    return mmds


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
