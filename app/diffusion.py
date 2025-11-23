import torch, math
import torch.nn as nn
from typing import Tuple, Optional
from helper_lib.diffusion_model import build_diffusion_model

def make_beta_schedule(T: int, beta_start=1e-4, beta_end=0.02, device="cpu"):
    betas = torch.linspace(beta_start, beta_end, T, device=device)
    alphas = 1.0 - betas
    alpha_bars = torch.cumprod(alphas, dim=0)
    return betas, alphas, alpha_bars

@torch.no_grad()
def q_sample(x0: torch.Tensor, t: torch.Tensor, alpha_bars: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    eps = torch.randn_like(x0)
    a_bar = alpha_bars[t].view(-1, 1, 1, 1)
    xt = torch.sqrt(a_bar) * x0 + torch.sqrt(1 - a_bar) * eps
    return xt, eps

def train_diffusion(model, data_loader, device: str = "cpu", epochs: int = 1, T: int = 1000, lr: float = 1e-3, max_batches: Optional[int] = None):
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    betas, alphas, alpha_bars = make_beta_schedule(T, device=device)
    for ep in range(1, epochs + 1):
        model.train(); b_idx = 0
        for x,_ in data_loader:
            b_idx += 1
            if max_batches and b_idx > max_batches: break
            x = x.to(device)
            B = x.size(0)
            t = torch.randint(0, T, (B,), device=device)
            xt, noise = q_sample(x, t, alpha_bars)
            pred = model(xt, t.float())
            loss = torch.mean((pred - noise)**2)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
        print(f"[Diffusion] Epoch {ep}/{epochs}")
    return model

@torch.no_grad()
def sample_ddpm(model, num_samples=16, img_size=32, img_channels=3, T=1000, device="cpu"):
    model.to(device).eval()
    betas, alphas, alpha_bars = make_beta_schedule(T, device=device)
    x = torch.randn(num_samples, img_channels, img_size, img_size, device=device)
    for ti in reversed(range(T)):
        t = torch.full((num_samples,), ti, device=device, dtype=torch.float32)
        eps_pred = model(x, t)
        alpha_t = alphas[ti]; beta_t = betas[ti]; alpha_bar_t = alpha_bars[ti]
        x = (1/torch.sqrt(alpha_t)) * (x - (beta_t/torch.sqrt(1-alpha_bar_t)) * eps_pred)
        if ti > 0:
            x = x + torch.sqrt(beta_t)*torch.randn_like(x)
    x = (x.clamp(-1,1) + 1)/2
    return x.cpu()

def load_or_init_diffusion(weights_path, device: str = "cpu", img_channels: int = 3):
    model = build_diffusion_model(img_channels=img_channels)
    if weights_path:
        try:
            sd = torch.load(weights_path, map_location=device)
            model.load_state_dict(sd)
            print(f"Loaded diffusion weights: {weights_path}")
        except FileNotFoundError:
            print("No diffusion weights found; fresh model.")
    return model
