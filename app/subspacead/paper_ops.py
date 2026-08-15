from __future__ import annotations

import logging

import cv2
import numpy as np
import torch
from tqdm import tqdm


class PCAModel:
    """Standard SubspaceAD PCA without importing scikit-learn.

    This mirrors the standard-PCA path used by the vendored SubspaceAD code:
    two streaming passes (mean, covariance), float64 eigendecomposition,
    explained-variance component selection, and the same saved parameter keys.
    KernelPCA is intentionally not included because it is not used by the
    paper-main / industrial SubspaceAD path.
    """

    def __init__(self, k=None, ev=None, whiten=False, eps=1e-6, device=None):
        self.k = k
        self.ev_ratio = ev
        self.whiten = whiten
        self.eps = eps
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.dtype = torch.float64
        self.mu_ = None
        self.components_ = None
        self.explained_variance_ = None
        self.eigvals_ = None
        self.pca_params = {}

    def _compute_mean(self, feature_generator, feature_dim, total_tokens, num_batches):
        logging.info("Starting PCA Pass 1/2 (Mean)...")
        self.mu_ = torch.zeros(feature_dim, dtype=self.dtype, device=self.device)
        for batch in tqdm(
            feature_generator(), total=num_batches, desc="PCA Pass 1/2 (Mean)"
        ):
            batch_gpu = torch.from_numpy(np.asarray(batch)).to(
                self.device, dtype=self.dtype
            )
            self.mu_ += torch.sum(batch_gpu, dim=0)
        self.mu_ /= total_tokens

    def _compute_covariance(self, feature_generator, feature_dim, total_tokens, num_batches):
        if total_tokens <= 1:
            raise ValueError("PCA requires at least two feature vectors")
        logging.info("Starting PCA Pass 2/2 (Covariance)...")
        cov_matrix = torch.zeros(
            (feature_dim, feature_dim), dtype=self.dtype, device=self.device
        )
        for batch in tqdm(
            feature_generator(), total=num_batches, desc="PCA Pass 2/2 (Cov)"
        ):
            batch_gpu = torch.from_numpy(np.asarray(batch)).to(
                self.device, dtype=self.dtype
            )
            centered = batch_gpu - self.mu_
            cov_matrix += centered.T @ centered
        cov_matrix /= total_tokens - 1
        return cov_matrix

    def _compute_eigendecomposition(self, cov_matrix):
        logging.info("Performing eigendecomposition...")
        evals, evecs = torch.linalg.eigh(cov_matrix)
        order = torch.argsort(evals, descending=True)
        self.explained_variance_ = evals[order]
        return evecs[:, order]

    def _select_k_components(self, evecs):
        if self.ev_ratio is not None and self.k is None:
            total = torch.sum(self.explained_variance_)
            if not torch.isfinite(total) or float(total.item()) <= 0.0:
                raise RuntimeError("PCA covariance has invalid total variance")
            cumulative = torch.cumsum(self.explained_variance_, dim=0) / total
            target = torch.tensor(
                [self.ev_ratio], dtype=self.dtype, device=self.device
            )
            self.k = int(torch.searchsorted(cumulative, target).item()) + 1

        if self.k is None:
            self.k = evecs.shape[1]
        else:
            self.k = min(int(self.k), evecs.shape[1])

        self.components_ = evecs[:, : self.k]
        self.eigvals_ = self.explained_variance_[: self.k]

    def _build_params(self):
        eigvals = self.eigvals_.detach().cpu().numpy().astype(np.float64)
        self.pca_params = {
            "mu": self.mu_.detach().cpu().numpy().astype(np.float64),
            "components": self.components_.detach().cpu().numpy().astype(np.float64),
            "eigvals": eigvals,
            "sqrt_eig": np.sqrt(eigvals + self.eps),
            "k": int(self.k),
            "whiten": bool(self.whiten),
            "eps": float(self.eps),
            "cov_Z_inv": np.diag(1.0 / (eigvals + self.eps)),
        }
        return self.pca_params

    def fit(self, feature_generator, feature_dim: int, total_tokens: int, num_batches: int):
        self._compute_mean(feature_generator, feature_dim, total_tokens, num_batches)
        cov = self._compute_covariance(
            feature_generator, feature_dim, total_tokens, num_batches
        )
        evecs = self._compute_eigendecomposition(cov)
        self._select_k_components(evecs)
        return self._build_params()


def pca_reconstruct(X: np.ndarray, pca: dict, drop_k: int = 0) -> np.ndarray:
    X = np.asarray(X)
    mu = np.asarray(pca["mu"], dtype=X.dtype)
    C = np.asarray(pca["components"][:, : pca["k"]], dtype=X.dtype)
    centered = X - mu
    Z = centered @ C
    if drop_k > 0:
        if drop_k >= Z.shape[1]:
            Z[:] = 0.0
        else:
            Z[:, :drop_k] = 0.0
    return (Z @ C.T) + mu


def calculate_anomaly_scores(
    X: np.ndarray,
    pca: dict,
    method: str = "reconstruction",
    drop_k: int = 0,
) -> np.ndarray:
    if method != "reconstruction":
        raise ValueError(
            "Industrial SubspaceAD paper path supports reconstruction scoring only"
        )
    if drop_k < 0:
        raise ValueError("drop_k must be non-negative")
    reconstructed = pca_reconstruct(X, pca, drop_k=drop_k)
    return np.sum((np.asarray(X) - reconstructed) ** 2, axis=1)


def post_process_map(
    anomaly_map: np.ndarray,
    res,
    blur: bool = True,
    close_holes: bool = False,
    close_k_size: int = 5,
):
    anomaly_map = np.asarray(anomaly_map, dtype=np.float32)
    dsize = (res, res) if isinstance(res, int) else (res[1], res[0])
    resized = cv2.resize(anomaly_map, dsize, interpolation=cv2.INTER_LINEAR)

    if close_holes:
        if close_k_size % 2 == 0:
            close_k_size += 1
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (close_k_size, close_k_size)
        )
        resized = cv2.morphologyEx(resized, cv2.MORPH_CLOSE, kernel)

    if blur:
        return cv2.GaussianBlur(resized, (3, 3), 4.0)
    return resized


def min_max_norm(x: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    x = np.nan_to_num(np.asarray(x), nan=0.0, posinf=0.0, neginf=0.0)
    x_min = np.min(x, axis=(-1, -2), keepdims=True)
    x_max = np.max(x, axis=(-1, -2), keepdims=True)
    return np.clip((x - x_min) / (x_max - x_min + eps), 0.0, 1.0)


def topk_mean(arr: np.ndarray, frac: float = 0.01) -> float:
    flat = np.asarray(arr).ravel()
    if flat.size == 0:
        raise ValueError("Cannot aggregate an empty anomaly map")
    k = max(1, int(flat.size * float(frac)))
    idx = np.argpartition(flat, -k)[-k:]
    return float(np.mean(flat[idx]))
