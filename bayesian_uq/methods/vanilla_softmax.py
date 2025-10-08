import time
import torch
import torch.nn.functional as F
from bayesian_uq.methods.base import BaseUQMethod, UQMethodConfig, UQPrediction
from bayesian_uq.methods import register_method
from bayesian_uq.utils import get_device, to_numpy


class VanillaSoftmaxConfig(UQMethodConfig):
    def __init__(self, **kwargs):
        defaults = {
            "name": "vanilla_softmax",
            "random_seed": 42,
            "device": "cpu",
            "verbose": True,
            "input_dim": 3,
            "hidden_dims": [128, 256, 128],
            "num_classes": 4,
            "dropout_rates": [0.0, 0.0, 0.0],
            "activation": "swish",
            "use_batch_norm": True,
            "use_layer_norm": False,
            "use_residual": True,
            "use_attention": False,
            "use_spectral_norm": False,
            "learning_rate": 0.001,
            "batch_size": 64,
            "max_epochs": 1000,
            "patience": 50,
            "min_delta": 0.0001,
            "weight_decay": 0.0001,
            "use_mixup": True,
            "mixup_alpha": 0.2,
            "label_smoothing": 0.1,
            "use_gradient_clipping": True,
            "max_grad_norm": 1.0,
            "optimizer_type": "adamw",
            "scheduler_type": "cosine_warmup",
            "warmup_epochs": 10,
        }
        defaults.update(kwargs)
        defaults["hidden_dims"] = list(defaults["hidden_dims"])
        defaults["dropout_rates"] = list(defaults["dropout_rates"])
        super().__init__(**defaults)
        if len(self.dropout_rates) != len(self.hidden_dims):
            self.dropout_rates = [0.0] * len(self.hidden_dims)


@register_method("vanilla_softmax")
class VanillaSoftmax(BaseUQMethod):
    config_class = VanillaSoftmaxConfig

    @staticmethod
    def default_config():
        return VanillaSoftmaxConfig()

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
        mlp_config.use_batch_norm = self.config.use_batch_norm
        mlp_config.use_layer_norm = self.config.use_layer_norm
        mlp_config.use_residual = self.config.use_residual
        mlp_config.use_attention = self.config.use_attention
        mlp_config.dropout_rates = list(self.config.dropout_rates)
        mlp_config.use_spectral_norm = self.config.use_spectral_norm
        mlp_config.label_smoothing = self.config.label_smoothing
        mlp_config.weight_decay = self.config.weight_decay
        mlp_config.learning_rate = self.config.learning_rate
        mlp_config.batch_size = self.config.batch_size
        mlp_config.max_epochs = self.config.max_epochs
        mlp_config.patience = self.config.patience
        mlp_config.min_delta = self.config.min_delta
        mlp_config.use_mixup = self.config.use_mixup
        mlp_config.mixup_alpha = self.config.mixup_alpha
        mlp_config.use_gradient_clipping = self.config.use_gradient_clipping
        mlp_config.max_grad_norm = self.config.max_grad_norm
        mlp_config.optimizer_type = self.config.optimizer_type
        mlp_config.scheduler_type = self.config.scheduler_type
        mlp_config.warmup_epochs = self.config.warmup_epochs
        mlp_config.uncertainty_method = "none"
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
        with torch.no_grad():
            logits = self.model(X_tensor)
            probs = F.softmax(logits, dim=1)
            predictions = to_numpy(probs.argmax(dim=1))
            probabilities = to_numpy(probs)
            entropy = -(probs * torch.log(probs.clamp_min(1e-08))).sum(dim=1)
            uncertainties = to_numpy(entropy)
            max_prob = probs.max(dim=1)[0]
            confidence = to_numpy(max_prob)
            logits_array = to_numpy(logits)
        inference_time = time.time() - start_time
        self.inference_time_per_sample = inference_time / len(X)
        return UQPrediction(
            predictions=predictions,
            probabilities=probabilities,
            uncertainties=uncertainties,
            aleatoric_uncertainty=None,
            epistemic_uncertainty=None,
            logits=logits_array,
            metadata={
                "method": "vanilla_softmax",
                "inference_time": inference_time,
                "confidence": confidence,
                "entropy_range": (uncertainties.min(), uncertainties.max()),
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
            from bayesian_uq.core.models import AdvancedMLP

            self.model = AdvancedMLP(self.mlp_config)
            self.model.to(self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.is_fitted = checkpoint.get("is_fitted", True)
