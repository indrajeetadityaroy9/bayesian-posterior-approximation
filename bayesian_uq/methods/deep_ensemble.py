import numpy as np
import torch
import torch.nn.functional as F
import time
from pathlib import Path
from bayesian_uq.methods.base import BaseUQMethod, UQMethodConfig, UQPrediction
from bayesian_uq.methods import register_method
from bayesian_uq.utils import get_device, to_numpy


class DeepEnsembleConfig(UQMethodConfig):
    def __init__(self, **kwargs):
        defaults = {
            "name": "deep_ensemble",
            "random_seed": 42,
            "device": "cpu",
            "verbose": True,
            "num_models": 5,
            "input_dim": 3,
            "hidden_dims": [128, 256, 128],
            "num_classes": 4,
            "dropout_rates": [0.1, 0.15, 0.1],
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
            self.dropout_rates = [0.1] * len(self.hidden_dims)


@register_method("deep_ensemble")
class DeepEnsemble(BaseUQMethod):
    config_class = DeepEnsembleConfig

    @staticmethod
    def default_config():
        return DeepEnsembleConfig()

    def __init__(self, config=None):
        if config is None:
            config = self.default_config()
        super().__init__(config)
        self.device = get_device(config.device)
        self.models = []
        self.trainers = []

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        from bayesian_uq.core.models import AdvancedMLPConfig
        from bayesian_uq.core.trainer import AdvancedTrainer

        start_time = time.time()
        self.models = []
        self.trainers = []
        all_metrics = []
        if self.config.verbose:
            print("=" * 70)
            print(f"Training Deep Ensemble with {self.config.num_models} models")
            print("=" * 70)
        for model_idx in range(self.config.num_models):
            if self.config.verbose:
                print(
                    f"\n[{model_idx + 1}/{self.config.num_models}] Training model {model_idx + 1}"
                )
            seed = self.config.random_seed + model_idx * 1000
            torch.manual_seed(seed)
            np.random.seed(seed)
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
            trainer = AdvancedTrainer(mlp_config)
            metrics_history = trainer.fit(X_train, y_train, X_val, y_val)
            trainer.model.eval()
            trainer.model.to(self.device)
            self.models.append(trainer.model)
            self.trainers.append(trainer)
            final_metrics = metrics_history[-1] if metrics_history else None
            final_val_accuracy = (
                float(getattr(final_metrics, "val_accuracy", 0.0))
                if final_metrics is not None
                else 0.0
            )
            final_val_loss = (
                float(getattr(final_metrics, "val_loss", 0.0))
                if final_metrics is not None
                else 0.0
            )
            if self.config.verbose:
                print(f"  Final validation accuracy: {final_val_accuracy:.2f}%")
            all_metrics.append(
                {
                    "model_id": model_idx,
                    "final_val_accuracy": final_val_accuracy,
                    "final_val_loss": final_val_loss,
                    "num_epochs": len(metrics_history),
                }
            )
        self.training_time = time.time() - start_time
        self.is_fitted = True
        avg_val_acc = (
            float(np.mean([m["final_val_accuracy"] for m in all_metrics]))
            if all_metrics
            else 0.0
        )
        avg_epochs = (
            int(np.mean([m["num_epochs"] for m in all_metrics])) if all_metrics else 0
        )
        return {
            "all_models_metrics": all_metrics,
            "final_val_accuracy": avg_val_acc,
            "average_val_accuracy": avg_val_acc,
            "std_val_accuracy": float(
                np.std([m["final_val_accuracy"] for m in all_metrics])
            )
            if all_metrics
            else 0.0,
            "num_epochs": avg_epochs,
            "training_time": self.training_time,
            "num_models": self.config.num_models,
        }

    def predict_with_uncertainty(self, X):
        if not self.is_fitted or not self.models:
            raise RuntimeError("DeepEnsemble must be fitted before prediction")
        start_time = time.time()
        X_tensor = torch.as_tensor(X, dtype=torch.float32, device=self.device)
        ensemble_probs = []
        with torch.no_grad():
            for model in self.models:
                logits = model(X_tensor)
                probs = F.softmax(logits, dim=1)
                ensemble_probs.append(probs)
        all_predictions = torch.stack(ensemble_probs, dim=0)
        mean_probs = all_predictions.mean(dim=0)
        aleatoric = (all_predictions * (1 - all_predictions)).mean(dim=0).sum(dim=1)
        epistemic = all_predictions.var(dim=0).sum(dim=1)
        total_uncertainty = aleatoric + epistemic
        predictions = to_numpy(mean_probs.argmax(dim=1))
        probabilities = to_numpy(mean_probs)
        uncertainties = to_numpy(total_uncertainty)
        aleatoric_np = to_numpy(aleatoric)
        epistemic_np = to_numpy(epistemic)
        logits = to_numpy(torch.log(mean_probs.clamp_min(1e-08)))
        inference_time = time.time() - start_time
        self.inference_time_per_sample = inference_time / max(len(X), 1)
        return UQPrediction(
            predictions=predictions,
            probabilities=probabilities,
            uncertainties=uncertainties,
            aleatoric_uncertainty=aleatoric_np,
            epistemic_uncertainty=epistemic_np,
            logits=logits,
            metadata={
                "num_models": self.config.num_models,
                "inference_time": inference_time,
                "method": "deep_ensemble",
            },
        )

    def get_num_parameters(self):
        if not self.models:
            return 0
        return sum(
            (sum((p.numel() for p in model.parameters())) for model in self.models)
        )

    def save(self, path):
        if not self.is_fitted or not self.models:
            raise RuntimeError("Cannot save an unfitted ensemble")
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        config_path = path.parent / f"{path.stem}_config.pt"
        torch.save(
            {
                "config": self.config.to_dict(),
                "num_models": len(self.models),
                "training_time": self.training_time,
                "is_fitted": self.is_fitted,
            },
            config_path,
        )
        for idx, model in enumerate(self.models):
            model_path = path.parent / f"{path.stem}_model_{idx}.pt"
            torch.save(
                {"model_state_dict": model.state_dict(), "model_id": idx}, model_path
            )

    def load(self, path):
        from bayesian_uq.core.models import AdvancedMLP, AdvancedMLPConfig

        path = Path(path)
        config_path = path.parent / f"{path.stem}_config.pt"
        checkpoint = torch.load(config_path, map_location=self.device)
        self.config = DeepEnsembleConfig(**checkpoint["config"])
        self.training_time = checkpoint.get("training_time", 0.0)
        self.is_fitted = checkpoint.get("is_fitted", True)
        num_models = int(checkpoint.get("num_models", 0))
        self.models = []
        for idx in range(num_models):
            model_path = path.parent / f"{path.stem}_model_{idx}.pt"
            model_checkpoint = torch.load(model_path, map_location=self.device)
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
            model = AdvancedMLP(mlp_config)
            model.load_state_dict(model_checkpoint["model_state_dict"])
            model.eval()
            model.to(self.device)
            self.models.append(model)
        self.trainers = []
