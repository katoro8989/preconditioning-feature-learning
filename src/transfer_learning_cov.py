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
    train_mode: str = "high"
    transfer_mode: str = "low"
    scale: float = 10.0
    snr: float = 1.0
    
    epochs: int = 10000
    lr: float = 1e-2
    wd: float = 1e-6
    exclude_bias: bool = True
    log_every: int = 1
    
    train_split: float = 0.02
    
    p: float = 0.0


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

@torch.no_grad()
def solve_head_ridge_closed_form(model, X_train, y_train, X_test=None, y_test=None, lam=1e-3, eps=1e-5):
    device = next(model.parameters()).device
    model.eval()

    H_tr = torch.relu(model.model[0](X_train)).detach()   # (n, d)
    n, d = H_tr.shape
    C = y_train.shape[1] if y_train.dim() == 2 else 1
    Y_tr = y_train.view(n, C).detach()

    h_mean = H_tr.mean(dim=0, keepdim=True)               # (1, d)
    y_mean = Y_tr.mean(dim=0, keepdim=True)               # (1, C)
    Hc = H_tr - h_mean
    Yc = Y_tr - y_mean

    Ht = Hc

    if d <= n:
        A = (Ht.T @ Ht) + (lam * n) * torch.eye(d, device=device)
        B = Ht.T @ Yc                                     # (d, C)
        Wc = torch.linalg.solve(A, B)                     # (d, C)
    else:
        A = (Ht @ Ht.T) + (lam * n) * torch.eye(n, device=device)
        alpha = torch.linalg.solve(A, Yc)                 # (n, C)
        Wc = Ht.T @ alpha                                 # (d, C)

    b = y_mean - h_mean @ Wc                              # (1, C)

    head = model.model[2]
    head.weight.copy_(Wc.T.contiguous())
    if head.bias is None:
        pass
    else:
        head.bias.copy_(b.view(-1))

    mse = nn.MSELoss()
    with torch.no_grad():
        y_hat_tr = torch.relu(model.model[0](X_train)) @ head.weight.T
        if head.bias is not None:
            y_hat_tr = y_hat_tr + head.bias
        train_mse = float(mse(y_hat_tr, Y_tr).item())
        test_mse = None
        if (X_test is not None) and (y_test is not None):
            Y_te = y_test.view(-1, C)
            H_te = torch.relu(model.model[0](X_test))
            y_hat_te = H_te @ head.weight.T
            if head.bias is not None:
                y_hat_te = y_hat_te + head.bias
            test_mse = float(mse(y_hat_te, Y_te).item())
    return model, train_mse, test_mse

def main():
    config = Config()

    if config.train_mode == "high":
        eigenvalues = np.array([config.scale]*config.k_components + [1 / config.scale]*(config.dx - config.k_components))
    elif config.train_mode == "low":
        eigenvalues = np.array([config.scale]*(config.dx - config.k_components) + [1 / config.scale]*config.k_components)
    eigenvectors = np.eye(config.dx)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    np.random.seed(config.seed); torch.manual_seed(config.seed)

    print("training")

    X_data, y_data, beta_star = generate_data_top_k_components(config.dx, config.n, eigenvalues, eigenvectors, k=config.k_components, snr=config.snr, mode=config.train_mode)

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

    model, tr1, te1 = train_with_p(init_model, config, eigenvectors, eigenvalues, X_train, y_train_n, X_test, y_test_n, device)

    X_data, y_data, beta_star = generate_data_top_k_components(config.dx, config.n, eigenvalues, eigenvectors, k=config.k_components, snr=config.snr, mode=config.transfer_mode)

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

    print("transfering")

    model, tr2, te2 = solve_head_ridge_closed_form(model, X_train, y_train_n, X_test, y_test_n)



    return tr1, te1, tr2, te2

if __name__ == "__main__":
    config = Config()
    tr1, te1, tr2, te2 = main()
    print(f"p={config.p:4.1f}: Train1 Loss = {tr1[-1]:.6f}, Test1 Loss = {te1[-1]:.6f}")
    print(f"p={config.p:4.1f}: Train2 Loss = {tr2:.6f}, Test2 Loss = {te2:.6f}")