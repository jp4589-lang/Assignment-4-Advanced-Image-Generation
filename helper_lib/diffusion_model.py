import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SinusoidalEmbedding(nn.Module):
    def __init__(self, dim: int = 64):
        super().__init__(); self.dim = dim
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        half = self.dim // 2
        freqs = torch.exp(torch.linspace(math.log(1.0), math.log(10000.0), half, device=t.device))
        x = t[:, None] / freqs[None, :]
        emb = torch.cat([torch.sin(x), torch.cos(x)], dim=-1)
        if self.dim % 2 == 1: emb = F.pad(emb, (0,1))
        return emb

class ConvBlock(nn.Module):
    def __init__(self, c_in, c_out):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(c_in, c_out, 3, padding=1), nn.SiLU(),
            nn.Conv2d(c_out, c_out, 3, padding=1), nn.SiLU(),
        )
    def forward(self, x): return self.net(x)

class UNetTiny(nn.Module):
    def __init__(self, img_channels=3, base=32, tdim=64):
        super().__init__()
        self.t_mlp = nn.Sequential(SinusoidalEmbedding(tdim), nn.Linear(tdim, tdim), nn.SiLU())
        self.enc1 = ConvBlock(img_channels, base)
        self.down1 = nn.Conv2d(base, base*2, 4, 2, 1)
        self.enc2 = ConvBlock(base*2, base*2)
        self.mid  = ConvBlock(base*2, base*2)
        self.up1  = nn.ConvTranspose2d(base*2, base, 4, 2, 1)
        self.dec1 = ConvBlock(base*2, base)
        self.out  = nn.Conv2d(base, img_channels, 3, padding=1)
        self.t_to_c1 = nn.Linear(tdim, base*2)
        self.t_to_c2 = nn.Linear(tdim, base)
    def forward(self, x, t):
        temb = self.t_mlp(t)
        e1 = self.enc1(x)
        d1 = self.down1(e1)
        e2 = self.enc2(d1)
        h  = self.mid(e2 + self.t_to_c1(temb).view(-1, e2.size(1), 1, 1))
        u1 = self.up1(h)
        cat = torch.cat([u1, e1 + self.t_to_c2(temb).view(-1, u1.size(1), 1, 1)], dim=1)
        dec = self.dec1(cat)
        return self.out(dec)

def build_diffusion_model(img_channels=3):
    return UNetTiny(img_channels=img_channels)
