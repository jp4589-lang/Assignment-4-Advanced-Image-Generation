import torch
import torch.nn as nn
from typing import Optional

class EnergyModel(nn.Module):
    def __init__(self, in_channels: int = 3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1), nn.SiLU(),
            nn.Conv2d(64, 128, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(128, 256, 4, stride=2, padding=1), nn.SiLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.SiLU(),
            nn.Conv2d(256, 1, 4, stride=2, padding=1),
        )
    def forward(self, x):
        h = self.net(x)
        return h.view(h.size(0), -1).mean(dim=1, keepdim=True)

@torch.no_grad()
def init_image(batch_size: int, channels: int, size: int, init: str = "noise", device: str = "cpu"):
    if init == "zeros":
        return torch.zeros(batch_size, channels, size, size, device=device, requires_grad=True)
    return torch.randn(batch_size, channels, size, size, device=device, requires_grad=True)

def sample_ebm(model: EnergyModel, steps: int = 200, lr: float = 0.05, batch_size: int = 16, channels: int = 3, size: int = 32, init: str = "noise", noise_scale: float = 0.01, device: str = "cpu"):
    model.to(device).eval()
    x = init_image(batch_size, channels, size, init=init, device=device)
    energies = []
    for s in range(steps):
        x.requires_grad_(True)
        energy = model(x).mean()
        grads = torch.autograd.grad(energy, x, create_graph=False)[0]
        with torch.no_grad():
            x = x - lr * grads
            if noise_scale > 0:
                x += noise_scale * torch.randn_like(x)
            x = x.clamp(-3, 3)
        energies.append(float(energy.item()))
    x = x.detach()
    minv, maxv = x.min(), x.max()
    x = (x - minv) / (maxv - minv + 1e-8)
    return x.cpu(), energies

def train_ebm(model: EnergyModel, data_loader, device: str = "cpu", epochs: int = 1, lr: float = 1e-4, neg_steps: int = 30, neg_lr: float = 0.1, noise_scale: float = 0.01, max_batches: Optional[int] = None):
    device = torch.device(device)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    margin = 1.0
    for ep in range(1, epochs + 1):
        running = 0.0; count = 0; b_idx = 0
        for real,_ in data_loader:
            b_idx += 1
            if max_batches and b_idx > max_batches: break
            real = real.to(device)
            E_pos = model(real).mean()
            negatives, _ = sample_ebm(model, steps=neg_steps, lr=neg_lr, batch_size=real.size(0), channels=real.size(1), size=real.size(2), noise_scale=noise_scale, device=str(device))
            negatives = negatives.to(device)
            E_neg = model(negatives).mean()
            loss = E_pos + torch.relu(margin - E_neg)
            opt.zero_grad(set_to_none=True)
            loss.backward(); opt.step()
            running += loss.item() * real.size(0); count += real.size(0)
        avg = running / max(count,1)
        print(f"[EBM] Epoch {ep}/{epochs} loss {avg:.4f}")
    return model
