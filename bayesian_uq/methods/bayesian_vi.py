import time
import torch
import torch.nn.functional as F
from bayesian_uq.methods.base import BaseUQMethod, UQMethodConfig, UQPrediction
from bayesian_uq.methods import register_method
from bayesian_uq.utils import get_device, to_numpy


class BayesianVIConfig(UQMethodConfig):
    def __init__(self, **kwargs):
        defaults = {
            "name": "bayesian_vi",
            "random_seed": 42,
            "device": "cpu",
            "verbose": True,
            "input_dim": 3,
            "hidden_dims": [128, 256, 128],
            "num_classes": 4,
            "activation": "swish",
            "learning_rate": 0.001,
            "batch_size": 64,
            "max_epochs": 1000,
            "patience": 50,
            "min_delta": 0.0001,
            "weight_decay": 0.0,
            "label_smoothing": 0.0,
            "use_gradient_clipping": True,
            "max_grad_norm": 1.0,
            "optimizer_type": "adamw",
            "scheduler_type": "cosine_warmup",
            "warmup_epochs": 10,
            "prior_std": 1.0,
            "vi_samples": 10,
            "kl_weight": 0.0005,
            "kl_anneal_epochs": 25,
        }
        defaults.update(kwargs)
        defaults["hidden_dims"] = list(defaults["hidden_dims"])
        super().__init__(**defaults)


@register_method("bayesian_vi")
class BayesianVI(BaseUQMethod):
    config_class = BayesianVIConfig

    @staticmethod
    def default_config():
        return BayesianVIConfig()

    def __init__(self, config=None):
        if config is None:
            config = self.default_config()
        super().__init__(config)
        self.device = get_device(self.config.device)
        from bayesian_uq.core.models import AdvancedMLPConfig

        mlp_config = AdvancedMLPConfig()
        mlp_config.input_dim = self.config.input_dim
        mlp_config.hidden_dims = list(self.config.hidden_dims)
        mlp_config.num_classes = self.config.num_classes
        mlp_config.activation = self.config.activation
        mlp_config.label_smoothing = self.config.label_smoothing
        mlp_config.weight_decay = self.config.weight_decay
        mlp_config.learning_rate = self.config.learning_rate
        mlp_config.batch_size = self.config.batch_size
        mlp_config.max_epochs = self.config.max_epochs
        mlp_config.patience = self.config.patience
        mlp_config.min_delta = self.config.min_delta
        mlp_config.use_gradient_clipping = self.config.use_gradient_clipping
        mlp_config.max_grad_norm = self.config.max_grad_norm
        mlp_config.optimizer_type = self.config.optimizer_type
        mlp_config.scheduler_type = self.config.scheduler_type
        mlp_config.warmup_epochs = self.config.warmup_epochs
        mlp_config.uncertainty_method = "bayesian"
        mlp_config.label_smoothing = 0.0
        mlp_config.weight_decay = 0.0
        mlp_config.prior_std = self.config.prior_std
        mlp_config.kl_weight = self.config.kl_weight
        mlp_config.kl_anneal_epochs = getattr(self.config, "kl_anneal_epochs", 0)
        mlp_config.vi_samples = self.config.vi_samples
        mlp_config.dropout_rates = [0.0] * len(mlp_config.hidden_dims)
        mlp_config.use_mixup = False
        self.mlp_config = mlp_config
        self.model = None
        self.trainer = None

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from bayesian_uq.core.trainer import AdvancedTrainer

        self.trainer = AdvancedTrainer(self.mlp_config)
        self.model = self.trainer.model
        metrics_history = self.trainer.fit(X_train, y_train, X_val, y_val)
        self.is_fitted = True
        final_metrics = metrics_history[-1]
        return {
            "final_train_loss": final_metrics.train_loss,
            "final_train_accuracy": final_metrics.train_accuracy,
            "final_val_loss": final_metrics.val_loss,
            "final_val_accuracy": final_metrics.val_accuracy,
            "num_epochs": final_metrics.epoch + 1,
        }

    def predict_with_uncertainty(self, X):
        if not self.is_fitted:
            raise RuntimeError("Model must be fitted before prediction")
        start_time = time.time()
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        all_probs = []
        with torch.no_grad():
            for _ in range(self.config.vi_samples):
                logits = self.model(X_tensor, sample=True)
                probs = F.softmax(logits, dim=1)
                all_probs.append(probs.cpu())
        all_probs = torch.stack(all_probs)
        mean_probs = all_probs.mean(dim=0)
        predictive_variance = all_probs.var(dim=0, unbiased=False)
        total_uncertainty = predictive_variance.sum(dim=1)
        entropy = -(mean_probs * torch.log(mean_probs.clamp_min(1e-08))).sum(dim=1)
        predictions = to_numpy(mean_probs.argmax(dim=1))
        probabilities = to_numpy(mean_probs)
        logits = to_numpy(torch.log(mean_probs.clamp_min(1e-08)))
        inference_time = time.time() - start_time
        self.inference_time_per_sample = inference_time / len(X)
        return UQPrediction(
            predictions=predictions,
            probabilities=probabilities,
            uncertainties=to_numpy(total_uncertainty),
            aleatoric_uncertainty=None,
            epistemic_uncertainty=to_numpy(total_uncertainty),
            logits=logits,
            metadata={
                "vi_samples": self.config.vi_samples,
                "prior_std": self.config.prior_std,
                "inference_time": inference_time,
                "method": "bayesian_vi",
                "predictive_entropy": to_numpy(entropy),
            },
        )

    def get_num_parameters(self):
        if self.model is None:
            return 0
        return sum((p.numel() for p in self.model.parameters()))

    def save(self, path):
        if not self.is_fitted:
            raise RuntimeError("Cannot save unfitted model")
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config.to_dict(),
                "is_fitted": self.is_fitted,
            },
            path,
        )

    def load(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        if self.model is None:
            from bayesian_uq.core.models import BayesianMLP

            self.model = BayesianMLP(self.mlp_config)
            self.model.to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.is_fitted = checkpoint.get("is_fitted", True)
