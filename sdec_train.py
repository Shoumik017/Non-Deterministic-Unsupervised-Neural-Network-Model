import os
import argparse
from dataclasses import dataclass

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

import numpy as np
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score, silhouette_score
from sklearn.cluster import KMeans


def get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")

def seed_everything(seed=42):
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


class Encoder(nn.Module):
    def __init__(self, d_in=784, d_hidden=500, d_latent=10, stochastic=True):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Linear(d_in, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
        )
        self.stochastic = stochastic
        self.mu = nn.Linear(d_hidden, d_latent)
        if stochastic:
            self.logvar = nn.Linear(d_hidden, d_latent)
        else:
            self.logvar = None

    def forward(self, x):
        h = self.backbone(x)
        mu = self.mu(h)
        if not self.stochastic:
            return mu, None, mu 
        logvar = self.logvar(h)
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        z = mu + std * eps
        return z, logvar, mu


class Decoder(nn.Module):
    def __init__(self, d_latent=10, d_hidden=500, d_out=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_latent, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_hidden), nn.ReLU(),
            nn.Linear(d_hidden, d_out),
        )
    def forward(self, z):
        return self.net(z)


@dataclass
class DECConfig:
    k: int = 10
    dof: float = 1.0  
    lambda_rec: float = 0.0  
    beta_kl: float = 1e-2   


def student_t_q(z, centers, v=1.0):
    dist2 = torch.cdist(z, centers) ** 2  # (B,K)
    num = (1.0 + dist2 / v) ** (-(v + 1.0) / 2.0)
    q = num / num.sum(dim=1, keepdim=True)
    return q

def make_target_p(q):
    f = (q ** 2) / q.sum(dim=0, keepdim=True).clamp_min(1e-9)
    p = f / f.sum(dim=1, keepdim=True).clamp_min(1e-9)
    return p

def kl_div_pq(p, q):
    return (p * (p.add(1e-12).log() - q.add(1e-12).log())).sum(dim=1).mean()

def latent_kl(mu, logvar):
    if logvar is None:
        return torch.tensor(0.0, device=mu.device)
    return -0.5 * (1 + logvar - mu.pow(2) - logvar.exp()).sum(dim=1).mean()


def pretrain_autoencoder(encoder, decoder, train_loader, device, epochs=5, lr=1e-3):
    enc_opt = torch.optim.Adam(encoder.parameters(), lr=lr)
    dec_opt = torch.optim.Adam(decoder.parameters(), lr=lr)
    mse = nn.MSELoss()
    encoder.train(); decoder.train()
    for ep in range(1, epochs+1):
        epoch_loss = 0.0
        for x, _ in train_loader:
            x = x.to(device).view(x.size(0), -1)
            z, _, _ = encoder(x)
            x_hat = decoder(z)
            loss = mse(x_hat, x)
            enc_opt.zero_grad(); dec_opt.zero_grad()
            loss.backward()
            enc_opt.step(); dec_opt.step()
            epoch_loss += loss.item() * x.size(0)
        print(f"[Pretrain] Epoch {ep}/{epochs}  MSE: {epoch_loss/len(train_loader.dataset):.4f}")
    return encoder, decoder

def init_centers_kmeans(encoder, data_loader, device, k=10):
    encoder.eval()
    feats = []
    with torch.no_grad():
        for x, _ in data_loader:
            x = x.to(device).view(x.size(0), -1)
            z, _, mu = encoder(x)
            rep = mu if encoder.stochastic else z
            feats.append(rep.detach().cpu().numpy())
    feats = np.concatenate(feats, axis=0)
    km = KMeans(n_clusters=k, n_init=20, random_state=42)
    km.fit(feats)
    centers = torch.tensor(km.cluster_centers_, dtype=torch.float32, device=device)
    return centers

def dec_finetune(encoder, decoder, centers, train_loader, cfg: DECConfig, device,
                 epochs=10, lr=1e-3, stochastic=True):
    params = list(encoder.parameters()) + [centers]
    if cfg.lambda_rec > 0.0:
        params += list(decoder.parameters())
    opt = torch.optim.Adam(params, lr=lr)
    mse = nn.MSELoss()
    encoder.train()
    if cfg.lambda_rec > 0.0:
        decoder.train()

    for ep in range(1, epochs+1):
        total = 0.0
        for x, _ in train_loader:
            x = x.to(device).view(x.size(0), -1)
            z, logvar, mu = encoder(x)
            q = student_t_q(z, centers, v=cfg.dof)
            p = make_target_p(q)
            loss = kl_div_pq(p, q)
            if cfg.lambda_rec > 0.0:
                x_hat = decoder(z if stochastic else mu)
                loss = loss + cfg.lambda_rec * mse(x_hat, x)
            if stochastic:
                loss = loss + cfg.beta_kl * latent_kl(mu, logvar)
            opt.zero_grad()
            loss.backward()
            opt.step()
            total += loss.item() * x.size(0)
        print(f"[DEC {'S' if stochastic else 'Det'}] Epoch {ep}/{epochs}  Loss: {total/len(train_loader.dataset):.4f}")
    return centers

def eval_metrics(encoder, centers, data_loader, device):
    encoder.eval()
    all_q = []
    all_rep = []
    all_labels = []
    with torch.no_grad():
        for x, y in data_loader:
            x = x.to(device).view(x.size(0), -1)
            z, _, mu = encoder(x)
            rep = (mu if encoder.stochastic else z).detach().cpu().numpy()
            q = student_t_q(z, centers).detach().cpu().numpy()
            all_rep.append(rep)
            all_q.append(q)
            all_labels.append(y.numpy())
    all_rep = np.concatenate(all_rep, axis=0)
    all_q = np.concatenate(all_q, axis=0)
    y_true = np.concatenate(all_labels, axis=0)
    y_pred = all_q.argmax(axis=1)
    ari = adjusted_rand_score(y_true, y_pred)
    nmi = normalized_mutual_info_score(y_true, y_pred)
    try:
        sil = silhouette_score(all_rep, y_pred)
    except Exception:
        sil = float("nan")
    return ari, nmi, sil


def make_loaders(batch=256):
    tfm = transforms.ToTensor()
    train_ds = datasets.MNIST(root="./data", train=True, download=True, transform=tfm)
    test_ds = datasets.MNIST(root="./data", train=False, download=True, transform=tfm)
    # Combine train+test for clustering; labels used only for metrics
    all_x = torch.cat([train_ds.data, test_ds.data], dim=0).float() / 255.0
    all_x = all_x.view(all_x.size(0), -1)
    all_y = torch.cat([train_ds.targets, test_ds.targets], dim=0)
    ds = TensorDataset(all_x, all_y)
    loader = DataLoader(ds, batch_size=batch, shuffle=True, num_workers=0)
    eval_loader = DataLoader(ds, batch_size=batch, shuffle=False, num_workers=0)
    return loader, eval_loader


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs_pre", type=int, default=5, help="AE pretrain epochs")
    parser.add_argument("--epochs_dec", type=int, default=10, help="DEC finetune epochs")
    parser.add_argument("--batch", type=int, default=256, help="batch size")
    parser.add_argument("--latent", type=int, default=10, help="latent dimension")
    parser.add_argument("--k", type=int, default=10, help="number of clusters")
    parser.add_argument("--lr", type=float, default=1e-3, help="learning rate")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = get_device()
    print("Device:", device)

    os.makedirs("./results", exist_ok=True)

    train_loader, eval_loader = make_loaders(batch=args.batch)

    print("\n=== Baseline (Deterministic DEC) ===")
    enc_det = Encoder(d_in=784, d_hidden=500, d_latent=args.latent, stochastic=False).to(device)
    dec_det = Decoder(d_latent=args.latent, d_hidden=500, d_out=784).to(device)
    pretrain_autoencoder(enc_det, dec_det, train_loader, device, epochs=args.epochs_pre, lr=args.lr)
    centers_det = init_centers_kmeans(enc_det, train_loader, device, k=args.k)
    cfg_det = DECConfig(k=args.k, dof=1.0, lambda_rec=0.0, beta_kl=0.0)
    dec_finetune(enc_det, dec_det, centers_det, train_loader, cfg_det, device,
                 epochs=args.epochs_dec, lr=args.lr, stochastic=False)
    det_ari, det_nmi, det_sil = eval_metrics(enc_det, centers_det, eval_loader, device)
    print(f"[Baseline] ARI={det_ari:.4f}  NMI={det_nmi:.4f}  Silhouette={det_sil:.4f}")

    print("\n=== S-DEC (Stochastic) ===")
    enc_sto = Encoder(d_in=784, d_hidden=500, d_latent=args.latent, stochastic=True).to(device)
    dec_sto = Decoder(d_latent=args.latent, d_hidden=500, d_out=784).to(device)
    pretrain_autoencoder(enc_sto, dec_sto, train_loader, device, epochs=args.epochs_pre, lr=args.lr)
    centers_sto = init_centers_kmeans(enc_sto, train_loader, device, k=args.k)
    cfg_sto = DECConfig(k=args.k, dof=1.0, lambda_rec=0.0, beta_kl=1e-2)
    dec_finetune(enc_sto, dec_sto, centers_sto, train_loader, cfg_sto, device,
                 epochs=args.epochs_dec, lr=args.lr, stochastic=True)
    sto_ari, sto_nmi, sto_sil = eval_metrics(enc_sto, centers_sto, eval_loader, device)
    print(f"[S-DEC]   ARI={sto_ari:.4f}  NMI={sto_nmi:.4f}  Silhouette={sto_sil:.4f}")

    with open("./results/metrics.txt", "w") as f:
        f.write("Model,ARI,NMI,Silhouette\n")
        f.write(f"Baseline,{det_ari:.6f},{det_nmi:.6f},{det_sil:.6f}\n")
        f.write(f"S-DEC,{sto_ari:.6f},{sto_nmi:.6f},{sto_sil:.6f}\n")
    print("\nSaved metrics to ./results/metrics.txt")
    print("Tip: rerun with different --seed values (e.g., 1..5) and report mean/std for stability.")

if __name__ == "__main__":
    main()