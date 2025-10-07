from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple
import numpy as np


@dataclass
class BenchmarkData:
    X_train: np.ndarray
    y_train: np.ndarray
    X_val: np.ndarray
    y_val: np.ndarray
    X_test: np.ndarray
    y_test: np.ndarray
    X_ood: Optional[np.ndarray] = None
    metadata: Dict = field(default_factory=dict)


@dataclass
class BenchmarkConfig:
    name: str = "benchmark"
    random_seed: int = 42
    train_size: int = 1000
    val_size: int = 200
    test_size: int = 10000
    ood_size: int = 1000
    difficulty: str = "medium"

    def updated(self, **kwargs):
        values = {**self.__dict__}
        values.update(kwargs)
        return type(self)(**values)


class BaseBenchmark:
    def __init__(self, config: Optional[BenchmarkConfig] = None, **overrides):
        default = self.default_config()
        if config is None:
            cfg = default
        elif isinstance(config, dict):
            cfg = default.updated(**config)
        elif isinstance(config, BenchmarkConfig):
            cfg = config
        else:
            raise TypeError("invalid benchmark config")
        if overrides:
            cfg = cfg.updated(**overrides)
        self.config = cfg
        self._data: Optional[BenchmarkData] = None
        self._bayes_error: Optional[float] = None

    @classmethod
    def default_config(cls) -> BenchmarkConfig:
        return BenchmarkConfig()

    def generate_data(self) -> BenchmarkData:
        rng = np.random.default_rng(self.config.random_seed)
        X_train, y_train = self._sample_split(rng, self.config.train_size)
        X_val, y_val = self._sample_split(rng, self.config.val_size)
        X_test, y_test = self._sample_split(rng, self.config.test_size)
        X_ood = (
            self._sample_ood(rng, self.config.ood_size)
            if self.config.ood_size
            else None
        )
        meta = self.metadata()
        data = BenchmarkData(
            X_train, y_train, X_val, y_val, X_test, y_test, X_ood, meta
        )
        self._data = data
        return data

    def metadata(self) -> Dict:
        return {
            "name": self.config.name,
            "num_classes": self.get_num_classes(),
            "input_dim": self.get_input_dim(),
            "train_size": self.config.train_size,
            "val_size": self.config.val_size,
            "test_size": self.config.test_size,
            "difficulty": self.config.difficulty,
        }

    def get_metadata(self) -> Dict:
        if self._data is None:
            return self.metadata()
        meta = dict(self.metadata())
        meta.update(self._data.metadata)
        return meta

    def get_bayes_error(self, sample_count: Optional[int] = None) -> float:
        if sample_count is not None:
            return self._estimate_bayes_error(sample_count)
        if self._bayes_error is None:
            self._bayes_error = self._estimate_bayes_error(None)
        return self._bayes_error

    def _estimate_bayes_error(self, sample_count: Optional[int]) -> float:
        size = sample_count or max(self.config.test_size, 10000)
        rng = np.random.default_rng(self.config.random_seed + 1)
        X, y = self._sample_split(rng, size)
        preds = self.compute_bayes_optimal(X)
        return float(np.mean(preds != y))

    def _sample_split(
        self, rng: np.random.Generator, size: int
    ) -> Tuple[np.ndarray, np.ndarray]:
        raise NotImplementedError

    def _sample_ood(self, rng: np.random.Generator, size: int) -> Optional[np.ndarray]:
        return None

    def compute_bayes_optimal(self, X: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def get_num_classes(self) -> int:
        raise NotImplementedError

    def get_input_dim(self) -> int:
        raise NotImplementedError
