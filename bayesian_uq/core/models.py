import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import math


class AdvancedMLPConfig:
    def __init__(self, **kwargs):
        defaults = {
            "input_dim": 3,
            "hidden_dims": [128, 256, 128],
            "num_classes": 4,
            "activation": "swish",
            "use_batch_norm": True,
            "use_layer_norm": False,
            "use_residual": True,
            "use_attention": False,
            "dropout_rates": None,
            "use_spectral_norm": False,
            "label_smoothing": 0.1,
            "weight_decay": 0.0001,
            "learning_rate": 0.001,
            "batch_size": 64,
            "max_epochs": 1000,
            "patience": 50,
            "min_delta": 0.0001,
            "use_mixup": True,
            "mixup_alpha": 0.2,
            "use_cutmix": False,
            "use_gradient_clipping": True,
            "max_grad_norm": 1.0,
            "optimizer_type": "adamw",
            "scheduler_type": "cosine_warmup",
            "warmup_epochs": 10,
            "uncertainty_method": "ensemble",
            "num_ensemble_models": 5,
            "mc_samples": 100,
            "vi_samples": 10,
            "prior_std": 1.0,
            "kl_weight": 1.0,
            "kl_anneal_epochs": 0,
        }
        defaults.update(kwargs)
        self.input_dim = defaults["input_dim"]
        self.hidden_dims = list(defaults["hidden_dims"])
        dropout_rates = defaults.get("dropout_rates")
        if dropout_rates is None:
            dropout_rates = [0.1] * len(self.hidden_dims)
        self.dropout_rates = list(dropout_rates)
        self.num_classes = defaults["num_classes"]
        self.activation = defaults["activation"]
        self.use_batch_norm = defaults["use_batch_norm"]
        self.use_layer_norm = defaults["use_layer_norm"]
        self.use_residual = defaults["use_residual"]
        self.use_attention = defaults["use_attention"]
        self.use_spectral_norm = defaults["use_spectral_norm"]
        self.label_smoothing = defaults["label_smoothing"]
        self.weight_decay = defaults["weight_decay"]
        self.learning_rate = defaults["learning_rate"]
        self.batch_size = defaults["batch_size"]
        self.max_epochs = defaults["max_epochs"]
        self.patience = defaults["patience"]
        self.min_delta = defaults["min_delta"]
        self.use_mixup = defaults["use_mixup"]
        self.mixup_alpha = defaults["mixup_alpha"]
        self.use_cutmix = defaults["use_cutmix"]
        self.use_gradient_clipping = defaults["use_gradient_clipping"]
        self.max_grad_norm = defaults["max_grad_norm"]
        self.optimizer_type = defaults["optimizer_type"]
        self.scheduler_type = defaults["scheduler_type"]
        self.warmup_epochs = defaults["warmup_epochs"]
        self.uncertainty_method = defaults["uncertainty_method"]
        self.num_ensemble_models = defaults["num_ensemble_models"]
        self.mc_samples = defaults["mc_samples"]
        self.vi_samples = defaults["vi_samples"]
        self.prior_std = defaults["prior_std"]
        self.kl_weight = defaults["kl_weight"]
        self.kl_anneal_epochs = defaults["kl_anneal_epochs"]


class AdvancedActivation(nn.Module):
    def __init__(self, activation_type="swish"):
        super().__init__()
        self.activation_type = activation_type.lower()
        if self.activation_type == "swish":
            self.activation = lambda x: x * torch.sigmoid(x)
        elif self.activation_type == "mish":
            self.activation = lambda x: x * torch.tanh(F.softplus(x))
        elif self.activation_type == "gelu":
            self.activation = F.gelu
        elif self.activation_type == "elu":
            self.activation = F.elu
        elif self.activation_type == "leaky_relu":
            self.activation = F.leaky_relu
        else:
            self.activation = F.relu

    def forward(self, x):
        return self.activation(x)


class AttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads=4):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        (attended, _) = self.attention(x, x, x)
        x = self.norm(x + attended)
        if x.size(1) == 1:
            x = x.squeeze(1)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        dropout_rate=0.1,
        activation="swish",
        use_batch_norm=True,
        use_layer_norm=False,
    ):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.activation = AdvancedActivation(activation)
        self.dropout = nn.Dropout(dropout_rate)
        self.norm = None
        if use_batch_norm:
            self.norm = nn.BatchNorm1d(out_dim)
        elif use_layer_norm:
            self.norm = nn.LayerNorm(out_dim)
        self.use_residual = in_dim == out_dim
        if not self.use_residual and in_dim != out_dim:
            self.residual_proj = nn.Linear(in_dim, out_dim)
        else:
            self.residual_proj = None

    def forward(self, x):
        identity = x
        out = self.linear(x)
        if self.norm is not None:
            out = self.norm(out)
        out = self.activation(out)
        out = self.dropout(out)
        if self.use_residual:
            out = out + identity
        elif self.residual_proj is not None:
            out = out + self.residual_proj(identity)
        return out


class BayesianLinear(nn.Module):
    def __init__(self, in_features, out_features, prior_std=1.0):
        super().__init__()
        self.weight_mu = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight_mu, a=math.sqrt(5))
        self.weight_rho = nn.Parameter(torch.full((out_features, in_features), -5.0))
        self.bias_mu = nn.Parameter(torch.empty(out_features))
        bound = 1 / math.sqrt(in_features) if in_features > 0 else 0.0
        nn.init.uniform_(self.bias_mu, -bound, bound)
        self.bias_rho = nn.Parameter(torch.full((out_features,), -5.0))
        self.prior_std = prior_std

    @property
    def weight_sigma(self):
        return F.softplus(self.weight_rho) + 1e-06

    @property
    def bias_sigma(self):
        return F.softplus(self.bias_rho) + 1e-06

    def forward(self, x, sample=True):
        if sample:
            weight_eps = torch.randn_like(self.weight_mu)
            bias_eps = torch.randn_like(self.bias_mu)
            weight = self.weight_mu + self.weight_sigma * weight_eps
            bias = self.bias_mu + self.bias_sigma * bias_eps
        else:
            weight = self.weight_mu
            bias = self.bias_mu
        return F.linear(x, weight, bias)

    def kl_divergence(self):
        weight_var = self.weight_sigma**2
        bias_var = self.bias_sigma**2
        kl_weight = 0.5 * torch.sum(
            self.weight_mu**2 / self.prior_std**2
            + weight_var / self.prior_std**2
            - torch.log(weight_var)
            + math.log(self.prior_std**2)
            - 1
        )
        kl_bias = 0.5 * torch.sum(
            self.bias_mu**2 / self.prior_std**2
            + bias_var / self.prior_std**2
            - torch.log(bias_var)
            + math.log(self.prior_std**2)
            - 1
        )
        return kl_weight + kl_bias


class AdvancedMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = []
        dims = [config.input_dim] + config.hidden_dims
        if config.use_attention:
            self.attention_block = AttentionBlock(config.input_dim)
        else:
            self.attention_block = None
        for i in range(len(config.hidden_dims)):
            layer = ResidualBlock(
                dims[i],
                dims[i + 1],
                dropout_rate=config.dropout_rates[i]
                if i < len(config.dropout_rates)
                else 0.1,
                activation=config.activation,
                use_batch_norm=config.use_batch_norm,
                use_layer_norm=config.use_layer_norm,
            )
            if config.use_spectral_norm:
                nn.utils.spectral_norm(layer.linear)
            layers.append(layer)
        self.hidden_layers = nn.ModuleList(layers)
        self.output_layer = nn.Linear(config.hidden_dims[-1], config.num_classes)
        if config.use_spectral_norm:
            nn.utils.spectral_norm(self.output_layer)
        self._initialize_weights()

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, (nn.BatchNorm1d, nn.LayerNorm)):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    def forward(self, x, return_features=False):
        features = []
        if self.attention_block is not None:
            x = self.attention_block(x)
        for layer in self.hidden_layers:
            x = layer(x)
            if return_features:
                features.append(x.clone())
        logits = self.output_layer(x)
        if return_features:
            return (logits, features)
        return logits


class BayesianMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        layers = []
        dims = [config.input_dim] + config.hidden_dims + [config.num_classes]
        for i in range(len(dims) - 1):
            layer = BayesianLinear(dims[i], dims[i + 1], prior_std=config.prior_std)
            layers.append(layer)
        self.layers = nn.ModuleList(layers)
        self.activation = AdvancedActivation(config.activation)

    def forward(self, x, sample=True):
        for i, layer in enumerate(self.layers):
            x = layer(x, sample=sample)
            if i < len(self.layers) - 1:
                x = self.activation(x)
        return x

    def kl_divergence(self):
        return sum((layer.kl_divergence() for layer in self.layers))


class MCDropoutMLP(AdvancedMLP):
    def __init__(self, config):
        super().__init__(config)

    def mc_forward(self, x, num_samples=100):
        previous_mode = self.training
        self.train(True)
        bn_modules = []
        for module in self.modules():
            if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
                bn_modules.append((module, module.training))
                module.eval()
        probs_samples = []
        with torch.no_grad():
            for _ in range(num_samples):
                logits = super().forward(x)
                probs_samples.append(F.softmax(logits, dim=1))
        probs_stack = torch.stack(probs_samples)
        mean_probs = probs_stack.mean(dim=0)
        predictive_variance = probs_stack.var(dim=0, unbiased=False)
        mean_log_probs = torch.log(mean_probs.clamp_min(1e-08))
        for module, state in bn_modules:
            module.train(state)
        self.train(previous_mode)
        return (mean_probs, predictive_variance, mean_log_probs)


def create_model(config):
    if config.uncertainty_method == "bayesian":
        return BayesianMLP(config)
    elif config.uncertainty_method == "mc_dropout":
        return MCDropoutMLP(config)
    else:
        return AdvancedMLP(config)


def mixup_data(x, y, alpha=0.2):
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index, :]
    (y_a, y_b) = (y, y[index])
    return (mixed_x, y_a, y_b, lam)


def mixup_criterion(criterion, pred, y_a, y_b, lam):
    return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


class TemperatureScaling(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model
        self.temperature = nn.Parameter(torch.ones(1))

    def forward(self, x):
        logits = self.model(x)
        return logits / self.temperature

    def set_temperature(self, valid_loader, device):
        self.model.eval()
        logits_list = []
        labels_list = []
        with torch.no_grad():
            for inputs, labels in valid_loader:
                (inputs, labels) = (inputs.to(device), labels.to(device))
                logits = self.model(inputs)
                logits_list.append(logits)
                labels_list.append(labels)
        logits = torch.cat(logits_list)
        labels = torch.cat(labels_list)
        optimizer = optim.LBFGS([self.temperature], lr=0.01, max_iter=50)

        def eval_loss():
            loss = F.cross_entropy(logits / self.temperature, labels)
            loss.backward()
            return loss

        optimizer.step(eval_loss)


if __name__ == "__main__":
    config = AdvancedMLPConfig(
        hidden_dims=[128, 256, 128],
        activation="swish",
        use_batch_norm=True,
        use_residual=True,
        uncertainty_method="ensemble",
    )
    model = create_model(config)
    x = torch.randn(32, 3)
    output = model(x)
    print("Model created successfully!")
    print(f"Input shape={x.shape}")
    print(f"Output shape={output.shape}")
    print(f"Model parameters={sum((p.numel() for p in model.parameters()))}")
