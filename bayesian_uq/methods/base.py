class UQPrediction:
    def __init__(
        self,
        predictions,
        probabilities,
        uncertainties,
        aleatoric_uncertainty=None,
        epistemic_uncertainty=None,
        logits=None,
        metadata=None,
    ):
        self.predictions = predictions
        self.probabilities = probabilities
        self.uncertainties = uncertainties
        self.aleatoric_uncertainty = aleatoric_uncertainty
        self.epistemic_uncertainty = epistemic_uncertainty
        self.logits = logits
        self.metadata = metadata or {}
        self._validate_shapes()

    def _validate_shapes(self):
        num_samples = self.probabilities.shape[0]
        if self.predictions.shape[0] != num_samples:
            raise ValueError("predictions and probabilities must share first dimension")
        if self.uncertainties.shape[0] != num_samples:
            raise ValueError(
                "uncertainties and probabilities must share first dimension"
            )
        if (
            self.aleatoric_uncertainty is not None
            and self.aleatoric_uncertainty.shape[0] != num_samples
        ):
            raise ValueError("aleatoric_uncertainty must align with probabilities")
        if (
            self.epistemic_uncertainty is not None
            and self.epistemic_uncertainty.shape[0] != num_samples
        ):
            raise ValueError("epistemic_uncertainty must align with probabilities")


class UQMethodConfig:
    def __init__(
        self,
        name="base_uq_method",
        random_seed=42,
        device="cpu",
        verbose=True,
        **kwargs,
    ):
        self.name = name
        self.random_seed = random_seed
        self.device = device
        self.verbose = verbose
        for key, value in kwargs.items():
            setattr(self, key, value)

    def to_dict(self):
        return self.__dict__.copy()

    @classmethod
    def from_dict(cls, config_dict):
        return cls(**config_dict)


class BaseUQMethod:
    def __init__(self, config):
        self.config = config
        self.is_fitted = False
        self.inference_time_per_sample = 0.0
        self._metadata = {}

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        raise NotImplementedError("fit() must be implemented by subclasses")

    def predict_with_uncertainty(self, X):
        raise NotImplementedError(
            "predict_with_uncertainty() must be implemented by subclasses"
        )

    def predict(self, X):
        prediction = self.predict_with_uncertainty(X)
        return prediction.predictions

    def predict_proba(self, X):
        prediction = self.predict_with_uncertainty(X)
        return prediction.probabilities

    def get_config(self):
        return {
            "method_name": self.__class__.__name__,
            "config": self.config.to_dict(),
            "is_fitted": self.is_fitted,
            "metadata": self._metadata,
        }

    def get_computational_cost(self):
        return {
            "inference_time_per_sample_ms": self.inference_time_per_sample * 1000,
            "total_parameters": self.get_num_parameters(),
        }

    def get_num_parameters(self):
        return 0

    def save(self, path):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement save()")

    def load(self, path):
        raise NotImplementedError(f"{self.__class__.__name__} does not implement load()")

    def _measure_inference_time(self, X, n_trials=10):
        raise NotImplementedError("Timing helpers require method-specific implementation")
