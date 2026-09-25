"""Select representative samples with PCA followed by k-medoids."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch


@dataclass(frozen=True)
class KMedoidsResult:
    """Results from :func:`pca_k_medoids`.

    ``medoids`` has the same array type and device as ``samples`` when the
    input is a NumPy array or PyTorch tensor.  The remaining arrays are NumPy
    arrays on the CPU.
    """

    medoid_indices: np.ndarray
    medoids: Any
    labels: np.ndarray
    embedding: np.ndarray
    explained_variance_ratio: np.ndarray
    inertia: float
    n_iter: int


def _as_numpy_samples(samples) -> tuple[np.ndarray, bool]:
    is_tensor = isinstance(samples, torch.Tensor)
    if is_tensor:
        cpu_samples = samples.detach().cpu()
        # NumPy does not directly support PyTorch's bfloat16 dtype.
        if cpu_samples.dtype == torch.bfloat16:
            cpu_samples = cpu_samples.float()
        values = cpu_samples.numpy()
    else:
        values = np.asarray(samples)
    if values.ndim < 2:
        raise ValueError("samples must have shape (n_samples, ...)")
    if values.shape[0] < 2:
        raise ValueError("at least two samples are required")
    if values[0].size == 0:
        raise ValueError("samples must contain at least one feature")
    if not np.issubdtype(values.dtype, np.number) or np.iscomplexobj(values):
        raise TypeError("samples must contain real numeric values")
    if not np.isfinite(values).all():
        raise ValueError("samples must contain only finite values")
    # PCA and Euclidean distance calculations readily overflow float32 for
    # valid physical quantities such as number densities around 1e20.
    return values.reshape(values.shape[0], -1).astype(np.float64, copy=False), is_tensor


def _validate_weights(weights, n_samples: int, n_medoids: int, dtype) -> np.ndarray:
    if weights is None:
        result = np.ones(n_samples, dtype=dtype)
    else:
        if isinstance(weights, torch.Tensor):
            weights = weights.detach().cpu().numpy()
        result = np.asarray(weights, dtype=dtype)
        if result.shape != (n_samples,):
            raise ValueError(f"weights must have shape ({n_samples},)")
        if not np.isfinite(result).all():
            raise ValueError("weights must contain only finite values")
        if np.any(result < 0):
            raise ValueError("weights must be nonnegative")
    if np.count_nonzero(result > 0) < n_medoids:
        raise ValueError("at least n_medoids samples must have positive weight")
    return result


def _randomized_pca(
    samples: np.ndarray,
    n_components: int,
    random_state: int | None,
) -> tuple[np.ndarray, np.ndarray]:
    centered = samples - samples.mean(axis=0, keepdims=True)
    max_components = min(centered.shape[0] - 1, centered.shape[1])
    n_components = min(n_components, max_components)

    # A small oversampled subspace and two power iterations give an accurate
    # truncated decomposition without constructing a feature covariance matrix.
    subspace_size = min(max_components, n_components + 10)
    rng = np.random.default_rng(random_state)
    projection = rng.standard_normal((centered.shape[1], subspace_size)).astype(centered.dtype)
    basis = centered @ projection
    for _ in range(2):
        basis, _ = np.linalg.qr(basis, mode="reduced")
        feature_basis, _ = np.linalg.qr(centered.T @ basis, mode="reduced")
        basis = centered @ feature_basis
    basis, _ = np.linalg.qr(basis, mode="reduced")

    small_matrix = basis.T @ centered
    left_vectors, singular_values, _ = np.linalg.svd(small_matrix, full_matrices=False)
    embedding = (basis @ left_vectors[:, :n_components]) * singular_values[:n_components]

    total_variance = np.square(centered).sum(dtype=np.float64)
    if total_variance == 0:
        explained_variance_ratio = np.zeros(n_components, dtype=centered.dtype)
    else:
        explained_variance_ratio = np.square(singular_values[:n_components]) / total_variance
    return embedding, explained_variance_ratio


def _pairwise_distances(values: np.ndarray) -> np.ndarray:
    # Scaling every coordinate by the same constant does not affect k-medoids,
    # and prevents avoidable overflow in the squared-distance calculation.
    scale = float(np.max(np.abs(values)))
    if scale == 0:
        return np.zeros((len(values), len(values)), dtype=values.dtype)
    scaled = values / scale
    squared_norms = np.einsum("ij,ij->i", scaled, scaled)
    squared_distances = squared_norms[:, None] + squared_norms[None, :] - 2 * scaled @ scaled.T
    np.maximum(squared_distances, 0, out=squared_distances)
    np.sqrt(squared_distances, out=squared_distances)
    squared_distances *= scale
    np.fill_diagonal(squared_distances, 0)
    if not np.isfinite(squared_distances).all():
        raise ValueError("PCA distance calculation produced non-finite values; consider rescaling the samples")
    return squared_distances


def _initial_medoids(distances: np.ndarray, weights: np.ndarray, n_medoids: int) -> np.ndarray:
    candidates = np.flatnonzero(weights > 0)
    first = candidates[np.argmin(weights @ distances[:, candidates])]
    medoids = [int(first)]
    nearest = distances[:, first].copy()

    # Greedily add the point that most reduces the weighted assignment cost.
    while len(medoids) < n_medoids:
        best_candidate = None
        best_cost = np.inf
        for candidate in candidates:
            if candidate in medoids:
                continue
            cost = float(weights @ np.minimum(nearest, distances[:, candidate]))
            if cost < best_cost:
                best_candidate = int(candidate)
                best_cost = cost
        if best_candidate is None:
            raise ValueError("k-medoids could not find a finite candidate cost")
        medoids.append(best_candidate)
        np.minimum(nearest, distances[:, best_candidate], out=nearest)
    return np.asarray(medoids, dtype=np.int64)


def _fit_k_medoids(
    distances: np.ndarray,
    weights: np.ndarray,
    n_medoids: int,
    max_iter: int,
) -> tuple[np.ndarray, np.ndarray, float, int]:
    medoids = _initial_medoids(distances, weights, n_medoids)
    positive = np.flatnonzero(weights > 0)

    for iteration in range(1, max_iter + 1):
        labels = np.argmin(distances[:, medoids], axis=1)
        updated = medoids.copy()
        for cluster in range(n_medoids):
            assigned = positive[labels[positive] == cluster]
            if assigned.size == 0:
                continue
            # A duplicate-valued medoid can be assigned to an earlier cluster
            # by argmin.  Do not let that cluster also adopt its index.
            other_medoids = np.delete(medoids, cluster)
            candidates = assigned[~np.isin(assigned, other_medoids)]
            if candidates.size == 0:
                continue
            costs = weights[assigned] @ distances[np.ix_(assigned, candidates)]
            updated[cluster] = candidates[np.argmin(costs)]
        if np.array_equal(updated, medoids):
            break
        medoids = updated

    labels = np.argmin(distances[:, medoids], axis=1).astype(np.int64, copy=False)
    inertia = float(weights @ distances[np.arange(len(weights)), medoids[labels]])
    return medoids, labels, inertia, iteration


def pca_k_medoids(
    samples,
    n_medoids: int,
    *,
    n_components: int = 16,
    weights=None,
    max_iter: int = 100,
    random_state: int | None = 0,
) -> KMedoidsResult:
    """Select representative samples using truncated PCA and k-medoids.

    The first dimension of ``samples`` is treated as the sample dimension and
    all remaining dimensions are flattened before PCA.  ``weights`` changes
    the clustering objective to ``sum(weights * distance_to_medoid)``; it does
    not change the PCA projection.  Zero-weight samples are assigned a cluster
    but cannot become medoids.

    The requested number of PCA components is automatically capped at the
    maximum rank, ``min(n_samples - 1, n_features)``.  Pairwise distances are
    stored explicitly, so memory use is quadratic in the number of samples.

    Args:
        samples: NumPy array, PyTorch tensor, or array-like batch.
        n_medoids: Number of representative samples to select.
        n_components: Requested size of the PCA embedding.
        weights: Optional nonnegative weight for each sample.
        max_iter: Maximum number of k-medoids refinement iterations.
        random_state: Seed for randomized PCA, or ``None`` for a random seed.

    Returns:
        A :class:`KMedoidsResult` containing representatives and diagnostics.
    """
    values, is_tensor = _as_numpy_samples(samples)
    n_samples = values.shape[0]
    if not isinstance(n_medoids, (int, np.integer)) or isinstance(n_medoids, bool):
        raise TypeError("n_medoids must be an integer")
    if not 1 <= n_medoids <= n_samples:
        raise ValueError("n_medoids must be between 1 and the number of samples")
    if not isinstance(n_components, (int, np.integer)) or isinstance(n_components, bool):
        raise TypeError("n_components must be an integer")
    if n_components <= 0:
        raise ValueError("n_components must be positive")
    if not isinstance(max_iter, (int, np.integer)) or isinstance(max_iter, bool):
        raise TypeError("max_iter must be an integer")
    if max_iter <= 0:
        raise ValueError("max_iter must be positive")

    validated_weights = _validate_weights(weights, n_samples, n_medoids, values.dtype)
    embedding, explained_variance_ratio = _randomized_pca(values, n_components, random_state)
    distances = _pairwise_distances(embedding)
    medoid_indices, labels, inertia, n_iter = _fit_k_medoids(
        distances,
        validated_weights,
        n_medoids,
        max_iter,
    )

    if is_tensor:
        index = torch.as_tensor(medoid_indices, device=samples.device)
        medoids = samples.index_select(0, index)
    else:
        medoids = np.asarray(samples)[medoid_indices]
    return KMedoidsResult(
        medoid_indices=medoid_indices,
        medoids=medoids,
        labels=labels,
        embedding=embedding,
        explained_variance_ratio=explained_variance_ratio,
        inertia=inertia,
        n_iter=n_iter,
    )
