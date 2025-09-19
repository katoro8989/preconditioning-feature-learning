import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from dataclasses import dataclass
import copy

@dataclass
class Config:
    dx: int = 10
    dh: int = 500
    dy: int = 1
    
    n: int = 10000
    k_components: int = 1
    seed: int = 42
    mode: str = "high"
    scale: float = 10.0
    snr: float = 1.0
    
    epochs: int = 10000
    lr: float = 1e-2
    wd: float = 1e-6
    exclude_bias: bool = True
    log_every: int = 1
    
    train_split: float = 0.02
    
    p: float = 0


def teacher_activation_and_derivatives(z):
    sigma_star = np.log(1 + np.exp(z*10))
    return sigma_star

def generate_data_top_k_components(dx, n, eigenvalues, eigenvectors, k=50, noise_std=1.0, snr=None, mode="high"):
    Sigma_x = eigenvectors @ np.diag(eigenvalues) @ eigenvectors.T
    if mode == "high":
      top_k_indices = np.argsort(eigenvalues)[-k:]
    elif mode == "low":
      top_k_indices = np.argsort(eigenvalues)[:k]
    top_k_eigenvectors = eigenvectors[:, top_k_indices]
    random_coeffs = np.random.randn(k, 1)
    beta_star = top_k_eigenvectors @ random_coeffs
    C = np.linalg.cholesky(Sigma_x)
    X = (C @ np.random.randn(dx, n)).T
    activations = X @ beta_star
    y_signal = teacher_activation_and_derivatives(activations)
    if snr is not None:
      signal_std = np.std(y_signal)
      noise_std = signal_std / snr
    epsilon = np.random.randn(n, 1) * noise_std
    y = y_signal + epsilon
    return X, y, beta_star


def generate_random_orthogonal_matrix(dim):
    H = np.random.randn(dim, dim)
    Q, R = np.linalg.qr(H)
    return Q

class MLP(nn.Module):
    def __init__(self, config: Config):
        super(MLP, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(config.dx, config.dh),
            nn.ReLU(),
            nn.Linear(config.dh, config.dy),
        )

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.model(x)

def init_small(m, scale=0.001):
    if isinstance(m, nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')
        m.weight.data *= scale
        if m.bias is not None:
            nn.init.constant_(m.bias, 0.0)

def make_model(config: Config):
    model = MLP(config)
    model.apply(lambda m: init_small(m, scale=.001))
    return model

def train_with_p(init_model, config: Config, eigenvectors, eigenvalues, X_train, y_train_n, X_test, y_test_n, device):
    model = copy.deepcopy(init_model).to(device)
    loss_fn = nn.MSELoss()

    P = eigenvectors @ np.diag(eigenvalues ** config.p) @ eigenvectors.T
    P_torch = torch.from_numpy(P.astype(np.float32)).to(device)

    train_hist, test_hist = [], []

    for ep in tqdm(range(config.epochs)):
        model.train()
        y_pred = model(X_train)
        data_loss = loss_fn(y_pred, y_train_n)

        reg = 0.0
        for name, param in model.named_parameters():
            if config.exclude_bias and 'bias' in name:
                continue
            reg = reg + param.pow(2).sum()
        loss = data_loss + 0.5 * config.wd * reg / X_train.shape[0]
        loss.backward()

        first_linear = model.model[0]
        first_linear.weight.grad = first_linear.weight.grad @ P_torch.T

        with torch.no_grad():
            for param in model.parameters():
                param -= config.lr * param.grad
        model.zero_grad(set_to_none=True)

        if (ep % config.log_every == 0) or (ep == config.epochs - 1):
            model.eval()
            with torch.no_grad():
                y_pred_test = model(X_test)
                test_loss = loss_fn(y_pred_test, y_test_n).item()
            train_hist.append(float(loss.item()))
            test_hist.append(test_loss)

    return model, np.array(train_hist), np.array(test_hist)


def main():
    config = Config()
    
    if config.mode == "high":
        eigenvalues = np.array([config.scale]*config.k_components + [1 / config.scale]*(config.dx - config.k_components))
    elif config.mode == "low":
        eigenvalues = np.array([config.scale]*(config.dx - config.k_components) + [1 / config.scale]*config.k_components)
    eigenvectors = np.eye(config.dx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(config.seed); torch.manual_seed(config.seed)

    X_data, y_data, beta_star = generate_data_top_k_components(config.dx, config.n, eigenvalues, eigenvectors, k=config.k_components, snr=config.snr, mode=config.mode)

    n = X_data.shape[0]
    perm = np.random.permutation(n)
    split = int(config.train_split * n)
    train_idx, test_idx = perm[:split], perm[split:]

    X_train_np, y_train_np = X_data[train_idx], y_data[train_idx]
    X_test_np,  y_test_np  = X_data[test_idx],  y_data[test_idx]

    X_train = torch.from_numpy(X_train_np).float().to(device)
    y_train = torch.from_numpy(y_train_np).float().to(device)
    X_test  = torch.from_numpy(X_test_np).float().to(device)
    y_test  = torch.from_numpy(y_test_np).float().to(device)

    y_mean = y_train.mean()
    y_std  = y_train.std()
    y_train_n = (y_train - y_mean) / (y_std + 1e-8)
    y_test_n  = (y_test  - y_mean) / (y_std + 1e-8)

    init_model = make_model(config).to(device)

    model, tr, te = train_with_p(init_model, config, eigenvectors, eigenvalues, X_train, y_train_n, X_test, y_test_n, device)

    return tr, te

if __name__ == "__main__":
    tr, te = main()
    print(f"Train Loss = {tr[-1]:.6f}, Test Loss = {te[-1]:.6f}")