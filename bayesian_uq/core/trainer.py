import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import time
import copy
import warnings
from bayesian_uq.core.models import (
    AdvancedMLPConfig,
    BayesianMLP,
    MCDropoutMLP,
    create_model,
    mixup_data,
    mixup_criterion,
)
from bayesian_uq.utils import get_device, to_numpy


class TrainingMetrics:
    def __init__(
        self,
        epoch,
        train_loss=None,
        train_accuracy=None,
        val_loss=None,
        val_accuracy=None,
        learning_rate=None,
        uncertainty_score=0.0,
        calibration_error=0.0,
        kl_loss=0.0,
    ):
        self.epoch = epoch
        self.train_loss = train_loss
        self.train_accuracy = train_accuracy
        self.val_loss = val_loss
        self.val_accuracy = val_accuracy
        self.learning_rate = learning_rate
        self.uncertainty_score = uncertainty_score
        self.calibration_error = calibration_error
        self.kl_loss = kl_loss


class EarlyStopping:
    def __init__(self, patience=50, min_delta=0.0001, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float("inf")
        self.best_model_state = None
        self.wait = 0
        self.stopped_epoch = 0

    def __call__(self, val_loss, model):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.wait = 0
            if self.restore_best:
                self.best_model_state = copy.deepcopy(model.state_dict())
        else:
            self.wait += 1
        if self.wait >= self.patience:
            self.stopped_epoch = len(self._losses) if hasattr(self, "_losses") else 0
            if self.restore_best and self.best_model_state is not None:
                model.load_state_dict(self.best_model_state)
            return True
        return False


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-06):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]["lr"]

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (
                self.total_epochs - self.warmup_epochs
            )
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + np.cos(np.pi * progress)
            )
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr
        return lr


class ModelEnsemble:
    def __init__(self, config):
        self.config = config
        self.models = []
        self.device = get_device()

    def add_model(self, model):
        model.eval()
        model.to(self.device)
        self.models.append(model)

    def predict(self, x):
        predictions = []
        with torch.no_grad():
            for model in self.models:
                logits = model(x.to(self.device))
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.cpu())
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        uncertainty = predictions.var(dim=0).sum(dim=1)
        return (mean_pred, uncertainty)

    def predict_with_decomposed_uncertainty(self, x):
        predictions = []
        with torch.no_grad():
            for model in self.models:
                logits = model(x.to(self.device))
                probs = F.softmax(logits, dim=1)
                predictions.append(probs.cpu())
        predictions = torch.stack(predictions)
        mean_pred = predictions.mean(dim=0)
        aleatoric = (predictions * (1 - predictions)).mean(dim=0).sum(dim=1)
        epistemic = predictions.var(dim=0).sum(dim=1)
        logits = torch.log(mean_pred.clamp_min(1e-08))
        return {
            "predictions": mean_pred,
            "aleatoric_uncertainty": aleatoric,
            "epistemic_uncertainty": epistemic,
            "total_uncertainty": aleatoric + epistemic,
            "logits": logits,
        }


class AdvancedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = get_device()
        self.metrics_history = []
        self.model = create_model(config)
        self.model.to(self.device)
        self.is_bayesian = isinstance(self.model, BayesianMLP)
        self.optimizer = self._create_optimizer()
        self.scheduler = WarmupCosineScheduler(
            self.optimizer, config.warmup_epochs, config.max_epochs
        )
        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
        self.early_stopping = EarlyStopping(config.patience, config.min_delta)
        print(f"Trainer initialized on {self.device}")
        print(f"Model parameters={sum((p.numel() for p in self.model.parameters()))}")

    def _kl_scale(self, epoch):
        if not self.is_bayesian:
            return 0.0
        anneal_epochs = getattr(self.config, "kl_anneal_epochs", 0) or 0
        if anneal_epochs <= 0:
            return 1.0
        return min(1.0, (epoch + 1) / float(anneal_epochs))

    def _create_optimizer(self):
        if self.config.optimizer_type.lower() == "adamw":
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999),
            )
        elif self.config.optimizer_type.lower() == "adam":
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
            )
        elif self.config.optimizer_type.lower() == "sgd":
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay,
            )
        else:
            warnings.warn(
                f"Unknown optimizer {self.config.optimizer_type}, using AdamW"
            )
            return optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0
        total_kl = 0.0
        dataset_size = len(train_loader.dataset)
        for batch_idx, (data, targets) in enumerate(train_loader):
            (data, targets) = (data.to(self.device), targets.to(self.device))
            if self.config.use_mixup and np.random.random() > 0.5:
                (mixed_data, targets_a, targets_b, lam) = mixup_data(
                    data, targets, self.config.mixup_alpha
                )
                outputs = self.model(mixed_data)
                loss = mixup_criterion(
                    self.criterion, outputs, targets_a, targets_b, lam
                )
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)
            if self.is_bayesian:
                kl_scale = self._kl_scale(epoch)
                kl_term = self.model.kl_divergence()
                kl_term = kl_term * self.config.kl_weight / dataset_size
                kl_term = kl_term * kl_scale
                loss += kl_term
                total_kl += float(kl_term.detach().cpu().item())
            self.optimizer.zero_grad()
            loss.backward()
            if self.config.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.max_grad_norm
                )
            self.optimizer.step()
            total_loss += loss.item()
            (_, predicted) = outputs.max(1)
            total += targets.size(0)
            if self.config.use_mixup and "mixed_data" in locals():
                correct += predicted.eq(targets).sum().item()
            else:
                correct += predicted.eq(targets).sum().item()
        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total
        avg_kl = total_kl / len(train_loader) if self.is_bayesian else 0.0
        return (avg_loss, accuracy, avg_kl)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        uncertainty_scores = []
        with torch.no_grad():
            for data, targets in val_loader:
                (data, targets) = (data.to(self.device), targets.to(self.device))
                if isinstance(self.model, MCDropoutMLP):
                    (probs, variance, log_probs) = self.model.mc_forward(
                        data, self.config.mc_samples
                    )
                    loss = self._loss_from_probs(probs, targets)
                    uncertainty_scores.extend(to_numpy(variance.sum(dim=1)))
                    outputs = probs
                elif isinstance(self.model, BayesianMLP):
                    num_samples = max(1, getattr(self.config, "vi_samples", 1))
                    logits_samples = [
                        self.model(data, sample=True) for _ in range(num_samples)
                    ]
                    logits_stack = torch.stack(logits_samples, dim=0)
                    mean_logits = logits_stack.mean(dim=0)
                    loss = self.criterion(mean_logits, targets)
                    probs = F.softmax(mean_logits, dim=1)
                    entropy = -(probs * probs.log()).sum(dim=1)
                    uncertainty_scores.extend(to_numpy(entropy))
                    outputs = mean_logits
                else:
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)
                    probs = F.softmax(outputs, dim=1)
                    entropy = -(probs * probs.log()).sum(dim=1)
                    uncertainty_scores.extend(to_numpy(entropy))
                total_loss += loss.item()
                (_, predicted) = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        avg_uncertainty = np.mean(uncertainty_scores)
        return (avg_loss, accuracy, avg_uncertainty)

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train.astype(int))
        if X_val is None or y_val is None:
            dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            (train_dataset, val_dataset) = random_split(dataset, [train_size, val_size])
        else:
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val.astype(int))
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)
        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_dataset, batch_size=self.config.batch_size, shuffle=False, num_workers=0
        )
        print(
            f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples"
        )
        for epoch in range(self.config.max_epochs):
            start_time = time.time()
            (train_loss, train_acc, kl_loss) = self.train_epoch(train_loader, epoch)
            (val_loss, val_acc, uncertainty) = self.validate(val_loader)
            current_lr = self.scheduler.step(epoch)
            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                learning_rate=current_lr,
                uncertainty_score=uncertainty,
                kl_loss=kl_loss,
            )
            self.metrics_history.append(metrics)
            if epoch % 10 == 0 or epoch == self.config.max_epochs - 1:
                epoch_time = time.time() - start_time
                kl_str = f", KL {kl_loss:.4f}" if self.is_bayesian else ""
                print(
                    f"Epoch {epoch:3d}: Loss {train_loss:.4f}/{val_loss:.4f}, "
                    f"Acc {train_acc:.2f}%/{val_acc:.2f}%, "
                    f"LR {current_lr:.2e}{kl_str}, Time {epoch_time:.1f}s"
                )
            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping at epoch {epoch}")
                break
        return self.metrics_history

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)
        with torch.no_grad():
            if isinstance(self.model, MCDropoutMLP):
                (probs, variance, log_probs) = self.model.mc_forward(
                    X_tensor, self.config.mc_samples
                )
                uncertainty = to_numpy(variance.sum(dim=1))
                logits = log_probs
            else:
                logits = self.model(X_tensor)
                probs = F.softmax(logits, dim=1)
                entropy = -(probs * probs.log()).sum(dim=1)
                uncertainty = to_numpy(entropy)
            predictions = to_numpy(probs.argmax(dim=1))
            probabilities = to_numpy(probs)
            logits_array = to_numpy(logits)
        return (predictions, probabilities, uncertainty, logits_array)

    def _loss_from_probs(self, probs, targets):
        probs = probs.clamp_min(1e-08)
        if self.config.label_smoothing > 0 and probs.size(1) > 1:
            smoothing = self.config.label_smoothing
            num_classes = probs.size(1)
            smooth_value = smoothing / (num_classes - 1)
            target_probs = torch.full_like(probs, smooth_value)
            target_probs.scatter_(1, targets.unsqueeze(1), 1 - smoothing)
            loss = -(target_probs * probs.log()).sum(dim=1).mean()
        else:
            loss = F.nll_loss(probs.log(), targets)
        return loss

    def get_metrics_df(self):
        import pandas as pd

        return pd.DataFrame([m.__dict__.copy() for m in self.metrics_history])

    def save_model(self, path):
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "config": self.config.__dict__.copy(),
                "metrics_history": [m.__dict__.copy() for m in self.metrics_history],
            },
            path,
        )

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint["model_state_dict"])
        if "metrics_history" in checkpoint:
            self.metrics_history = [
                TrainingMetrics(**m) for m in checkpoint["metrics_history"]
            ]


def train_ensemble(X_train, y_train, X_val, y_val, config, num_models=5):
    ensemble = ModelEnsemble(config)
    for i in range(num_models):
        print(f"\nTraining ensemble model {i + 1}/{num_models}")
        model_config = copy.deepcopy(config)
        torch.manual_seed(42 + i * 1000)
        np.random.seed(42 + i * 1000)
        trainer = AdvancedTrainer(model_config)
        trainer.fit(X_train, y_train, X_val, y_val)
        ensemble.add_model(trainer.model)
        val_acc = trainer.metrics_history[-1].val_accuracy
        print(f"Model {i + 1} final validation accuracy={val_acc:.2f}%")
    return ensemble


if __name__ == "__main__":
    config = AdvancedMLPConfig(
        hidden_dims=[128, 256, 128], max_epochs=100, batch_size=32, learning_rate=0.001
    )
    X_train = np.random.randn(1000, 3)
    y_train = np.random.randint(0, 4, 1000)
    X_val = np.random.randn(200, 3)
    y_val = np.random.randint(0, 4, 200)
    trainer = AdvancedTrainer(config)
    metrics = trainer.fit(X_train, y_train, X_val, y_val)
    print("Training completed!")
    print(f"Final validation accuracy={metrics[-1].val_accuracy:.2f}%")
