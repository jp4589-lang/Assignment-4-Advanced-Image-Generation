import torch
import torch.nn as nn
from .diffusion_model import build_diffusion_model

class Generator(nn.Module):
    def __init__(self, z_dim: int = 100):
        super().__init__()
        self.net = nn.Sequential(
            nn.ConvTranspose2d(z_dim, 256, 4, 1, 0, bias=False), nn.BatchNorm2d(256), nn.ReLU(True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False), nn.BatchNorm2d(128), nn.ReLU(True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(True),
            nn.ConvTranspose2d(64, 1, 4, 2, 1, bias=False), nn.Tanh()
        )
    def forward(self, z):
        if z.dim()==2: z = z.view(z.size(0), z.size(1), 1, 1)
        return self.net(z)

class MNISTDiscriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(1,64,4,2,1), nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64,128,4,2,1), nn.BatchNorm2d(128), nn.LeakyReLU(0.2, inplace=True),
            nn.Flatten(), nn.Linear(128*7*7,1)
        )
    def forward(self,x): return self.net(x)

class MNISTGenerator(Generator):
    pass

class MNISTCritic(MNISTDiscriminator):
    pass

def get_model(name: str, **kwargs):
    name = name.lower()
    if name in {"mnist_gan", "gan_mnist"}:
        z_dim = kwargs.get("z_dim", 100)
        return {"gen": MNISTGenerator(z_dim), "critic": MNISTCritic(), "z_dim": z_dim}
    if name in {"diffusion", "ddpm"}:
        ch = kwargs.get("img_channels", 3)
        return build_diffusion_model(img_channels=ch)
    raise ValueError(f"Unknown model name: {name}")
