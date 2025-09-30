import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset, random_split
import numpy as np
import time
import copy
from dataclasses import dataclass, asdict
import warnings

from sota_mlp import (
    AdvancedMLPConfig, AdvancedMLP, BayesianMLP, MCDropoutMLP,
    create_model, mixup_data, mixup_criterion, TemperatureScaling
)

@dataclass
class TrainingMetrics:
    epoch: int
    train_loss: float
    train_accuracy: float
    val_loss: float
    val_accuracy: float
    learning_rate: float
    uncertainty_score: float = 0.0
    calibration_error: float = 0.0


class EarlyStopping:
    def __init__(self, patience=50, min_delta=1e-4, restore_best=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best = restore_best
        self.best_loss = float('inf')
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
            self.stopped_epoch = len(self._losses) if hasattr(self, '_losses') else 0
            if self.restore_best and self.best_model_state is not None:
                model.load_state_dict(self.best_model_state)
            return True
        return False


class WarmupCosineScheduler:
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr=1e-6):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr = min_lr
        self.base_lr = optimizer.param_groups[0]['lr']

    def step(self, epoch):
        if epoch < self.warmup_epochs:
            lr = self.base_lr * (epoch + 1) / self.warmup_epochs
        else:
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (1 + np.cos(np.pi * progress))

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr

        return lr


class ModelEnsemble:
    def __init__(self, config):
        self.config = config
        self.models = []
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        return mean_pred, uncertainty

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

        return {
            'predictions': mean_pred,
            'aleatoric_uncertainty': aleatoric,
            'epistemic_uncertainty': epistemic,
            'total_uncertainty': aleatoric + epistemic
        }


class AdvancedTrainer:
    def __init__(self, config):
        self.config = config
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metrics_history = []

        self.model = create_model(config)
        self.model.to(self.device)

        self.optimizer = self._create_optimizer()

        self.scheduler = WarmupCosineScheduler(
            self.optimizer,
            config.warmup_epochs,
            config.max_epochs
        )

        self.criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)

        self.early_stopping = EarlyStopping(config.patience, config.min_delta)

        print(f"Trainer initialized on {self.device}")
        print(f"Model parameters: {sum(p.numel() for p in self.model.parameters())}")

    def _create_optimizer(self):
        if self.config.optimizer_type.lower() == 'adamw':
            return optim.AdamW(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay,
                betas=(0.9, 0.999)
            )
        elif self.config.optimizer_type.lower() == 'adam':
            return optim.Adam(
                self.model.parameters(),
                lr=self.config.learning_rate,
                weight_decay=self.config.weight_decay
            )
        elif self.config.optimizer_type.lower() == 'sgd':
            return optim.SGD(
                self.model.parameters(),
                lr=self.config.learning_rate,
                momentum=0.9,
                weight_decay=self.config.weight_decay
            )
        else:
            warnings.warn(f"Unknown optimizer {self.config.optimizer_type}, using AdamW")
            return optim.AdamW(self.model.parameters(), lr=self.config.learning_rate)

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(self.device), targets.to(self.device)

            if self.config.use_mixup and np.random.random() > 0.5:
                mixed_data, targets_a, targets_b, lam = mixup_data(
                    data, targets, self.config.mixup_alpha
                )
                outputs = self.model(mixed_data)
                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)
            else:
                outputs = self.model(data)
                loss = self.criterion(outputs, targets)

            if isinstance(self.model, BayesianMLP):
                kl_loss = self.model.kl_divergence() / len(train_loader.dataset)
                loss += kl_loss

            self.optimizer.zero_grad()
            loss.backward()

            if self.config.use_gradient_clipping:
                torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(),
                    self.config.max_grad_norm
                )

            self.optimizer.step()

            total_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)

            if self.config.use_mixup and 'mixed_data' in locals():
                correct += predicted.eq(targets).sum().item()
            else:
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(train_loader)
        accuracy = 100.0 * correct / total

        return avg_loss, accuracy

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0
        uncertainty_scores = []

        with torch.no_grad():
            for data, targets in val_loader:
                data, targets = data.to(self.device), targets.to(self.device)

                if isinstance(self.model, MCDropoutMLP):
                    outputs, uncertainty = self.model.mc_forward(data, self.config.mc_samples)
                    loss = self.criterion(outputs.log(), targets)
                    uncertainty_scores.extend(uncertainty.sum(dim=1).cpu().numpy())
                else:
                    outputs = self.model(data)
                    loss = self.criterion(outputs, targets)

                    probs = F.softmax(outputs, dim=1)
                    entropy = -(probs * probs.log()).sum(dim=1)
                    uncertainty_scores.extend(entropy.cpu().numpy())

                total_loss += loss.item()
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()

        avg_loss = total_loss / len(val_loader)
        accuracy = 100.0 * correct / total
        avg_uncertainty = np.mean(uncertainty_scores)

        return avg_loss, accuracy, avg_uncertainty

    def fit(self, X_train, y_train, X_val=None, y_val=None):
        X_train_tensor = torch.FloatTensor(X_train)
        y_train_tensor = torch.LongTensor(y_train.astype(int))

        if X_val is None or y_val is None:
            dataset = TensorDataset(X_train_tensor, y_train_tensor)
            train_size = int(0.9 * len(dataset))
            val_size = len(dataset) - train_size
            train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
        else:
            train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
            X_val_tensor = torch.FloatTensor(X_val)
            y_val_tensor = torch.LongTensor(y_val.astype(int))
            val_dataset = TensorDataset(X_val_tensor, y_val_tensor)

        train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=0
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
            num_workers=0
        )

        print(f"Training on {len(train_dataset)} samples, validating on {len(val_dataset)} samples")

        for epoch in range(self.config.max_epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch(train_loader)

            val_loss, val_acc, uncertainty = self.validate(val_loader)

            current_lr = self.scheduler.step(epoch)

            metrics = TrainingMetrics(
                epoch=epoch,
                train_loss=train_loss,
                train_accuracy=train_acc,
                val_loss=val_loss,
                val_accuracy=val_acc,
                learning_rate=current_lr,
                uncertainty_score=uncertainty
            )
            self.metrics_history.append(metrics)

            if epoch % 10 == 0 or epoch == self.config.max_epochs - 1:
                epoch_time = time.time() - start_time
                print(f"Epoch {epoch:3d}: "
                      f"Loss {train_loss:.4f}/{val_loss:.4f}, "
                      f"Acc {train_acc:.2f}%/{val_acc:.2f}%, "
                      f"LR {current_lr:.2e}, "
                      f"Time {epoch_time:.1f}s")

            if self.early_stopping(val_loss, self.model):
                print(f"Early stopping at epoch {epoch}")
                break

        return self.metrics_history

    def predict(self, X):
        self.model.eval()
        X_tensor = torch.FloatTensor(X).to(self.device)

        with torch.no_grad():
            if isinstance(self.model, MCDropoutMLP):
                probs, uncertainty = self.model.mc_forward(X_tensor, self.config.mc_samples)
                uncertainty = uncertainty.sum(dim=1).cpu().numpy()
            else:
                logits = self.model(X_tensor)
                probs = F.softmax(logits, dim=1)

                entropy = -(probs * probs.log()).sum(dim=1)
                uncertainty = entropy.cpu().numpy()

            predictions = probs.argmax(dim=1).cpu().numpy()
            probabilities = probs.cpu().numpy()

        return predictions, probabilities, uncertainty

    def get_metrics_df(self):
        import pandas as pd
        return pd.DataFrame([asdict(m) for m in self.metrics_history])

    def save_model(self, path):
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'config': asdict(self.config),
            'metrics_history': [asdict(m) for m in self.metrics_history]
        }, path)

    def load_model(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'])
        if 'metrics_history' in checkpoint:
            self.metrics_history = [TrainingMetrics(**m) for m in checkpoint['metrics_history']]


def train_ensemble(X_train, y_train, X_val, y_val, config, num_models=5):
    ensemble = ModelEnsemble(config)

    for i in range(num_models):
        print(f"\nTraining ensemble model {i+1}/{num_models}")

        model_config = copy.deepcopy(config)
        torch.manual_seed(42 + i * 1000)
        np.random.seed(42 + i * 1000)

        trainer = AdvancedTrainer(model_config)
        trainer.fit(X_train, y_train, X_val, y_val)

        ensemble.add_model(trainer.model)

        print(f"Model {i+1} final validation accuracy: {trainer.metrics_history[-1].val_accuracy:.2f}%")

    return ensemble


if __name__ == "__main__":
    config = AdvancedMLPConfig(
        hidden_dims=[128, 256, 128],
        max_epochs=100,
        batch_size=32,
        learning_rate=1e-3
    )

    X_train = np.random.randn(1000, 3)
    y_train = np.random.randint(0, 4, 1000)
    X_val = np.random.randn(200, 3)
    y_val = np.random.randint(0, 4, 200)

    trainer = AdvancedTrainer(config)
    metrics = trainer.fit(X_train, y_train, X_val, y_val)

    print(f"Training completed!")
    print(f"Final validation accuracy: {metrics[-1].val_accuracy:.2f}%")