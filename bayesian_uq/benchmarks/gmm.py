from dataclasses import dataclass
from typing import Optional, Tuple
import numpy as np
from scipy.stats import multivariate_normal
from .base import BaseBenchmark, BenchmarkConfig


@dataclass
class GMMBenchmarkConfig(BenchmarkConfig):
    name: str = "gmm_4class"
    train_size: int = 1000
    val_size: int = 200
    test_size: int = 10000
    ood_size: int = 1000
    difficulty: str = "medium"
    use_default_params: bool = True
    priors: Optional[np.ndarray] = None
    mean_vectors: Optional[np.ndarray] = None
    cov_matrices: Optional[np.ndarray] = None

    def updated(self, **kwargs):
        return super().updated(**kwargs)


class GMMBenchmark(BaseBenchmark):
    @classmethod
    def default_config(cls) -> BenchmarkConfig:
        return GMMBenchmarkConfig()

    def __init__(self, config: Optional[BenchmarkConfig] = None, **overrides):
        super().__init__(config, **overrides)
        if self.config.use_default_params or self.config.priors is None:
            self._priors = np.array([0.25, 0.25, 0.25, 0.25])
            self._means = np.array(
                [[0.0, 0.0, 0.0], [2.5, 0.0, 0.0], [5.0, 0.0, 0.0], [7.5, 0.0, 0.0]]
            )
            self._covs = np.array(
                [
                    [[1.0, 0.3, 1.4], [0.3, 1.0, 0.3], [1.4, 0.3, 7.0]],
                    [[1.0, -0.4, -0.7], [-0.4, 1.0, -0.4], [-0.7, -0.4, 3.0]],
                    [[1.0, 0.4, 0.7], [0.4, 1.0, 0.4], [0.7, 0.4, 3.0]],
                    [[1.0, -0.3, -1.4], [-0.3, 1.0, -0.3], [-1.4, -0.3, 7.0]],
                ]
            )
        else:
            self._priors = np.asarray(self.config.priors)
            self._means = np.asarray(self.config.mean_vectors)
            self._covs = np.asarray(self.config.cov_matrices)
        self._distributions = [
            multivariate_normal(mean=self._means[i], cov=self._covs[i])
            for i in range(len(self._priors))
        ]

    def _sample_split(
        self, rng: np.random.Generator, size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        classes = rng.choice(len(self._priors), size=size, p=self._priors)
        X = np.empty((size, self._means.shape[1]))
        for idx in range(len(self._priors)):
            mask = classes == idx
            count = int(mask.sum())
            if count:
                X[mask] = rng.multivariate_normal(
                    self._means[idx], self._covs[idx], size=count
                )
        return X, classes.astype(int)

    def _sample_ood(self, rng: np.random.Generator, size: int) -> Optional[np.ndarray]:
        if size <= 0:
            return None
        classes = rng.integers(0, len(self._priors), size=size)
        X = np.empty((size, self._means.shape[1]))
        shift = np.array([10.0, 5.0, 5.0])
        for idx in range(len(self._priors)):
            mask = classes == idx
            count = int(mask.sum())
            if count:
                X[mask] = rng.multivariate_normal(
                    self._means[idx] + shift, self._covs[idx] * 2.0, size=count
                )
        return X

    def compute_bayes_optimal(self, X: np.ndarray) -> np.ndarray:
        posteriors = self._posterior(X)
        return np.argmax(posteriors, axis=1)

    def _posterior(self, X: np.ndarray) -> np.ndarray:
        densities = np.stack([dist.pdf(X) for dist in self._distributions], axis=1)
        weighted = densities * self._priors
        totals = weighted.sum(axis=1, keepdims=True)
        totals[totals == 0.0] = 1e-12
        return weighted / totals

    def get_num_classes(self) -> int:
        return len(self._priors)

    def get_input_dim(self) -> int:
        return int(self._means.shape[1])

    def metadata(self) -> dict:
        meta = super().metadata()
        meta.update({"priors": self._priors.tolist()})
        return meta
