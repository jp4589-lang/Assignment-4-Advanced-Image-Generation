# FastAPI service exposing Diffusion + EBM (CIFAR-10) and a minimal GAN sample endpoint.
# Limited training (2–3 epochs, capped batches) chosen to prioritize deployment; full-quality
# training would require 10+ hours of compute.
from typing import Optional
from fastapi import FastAPI, Response
from pydantic import BaseModel
from PIL import Image
import torch, base64, math, os
from io import BytesIO
from pathlib import Path
from torchvision.utils import make_grid, save_image
from torchvision import transforms, datasets
from torch.utils.data import DataLoader

from .diffusion import load_or_init_diffusion, sample_ddpm, train_diffusion
from .ebm import EnergyModel, sample_ebm, train_ebm
from helper_lib.model import get_model

app = FastAPI()

# Determine repo root (works both locally and in Docker)
_here = Path(__file__).resolve()
_env_root = os.getenv("REPO_ROOT")
_repo_root = Path(_env_root).resolve() if _env_root else _here.parents[1]
_models_root = _repo_root / "models"
_models_root.mkdir(parents=True, exist_ok=True)
DIFFUSION_WEIGHTS = _models_root / "diffusion_cifar10.pt"
EBM_WEIGHTS = _models_root / "ebm_cifar10.pt"

_device = (
    "cuda" if torch.cuda.is_available() else
    ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
)

def device():
    return _device

_gan_gen = None
_diffusion_model = None
_ebm_model = None

# ---------------- GAN (MNIST demo) ----------------
def _gan_weights_path() -> Path:
    candidates = [
        _repo_root / "outputs/gan_mnist.pt",
        Path("outputs/gan_mnist.pt"),
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("GAN weights not found: outputs/gan_mnist.pt. Train separately if needed.")

def _load_gan(z_dim: int = 100):
    global _gan_gen
    if _gan_gen is not None:
        return _gan_gen
    bundle = get_model("mnist_gan", z_dim=z_dim)
    gen = bundle["gen"].to(device()).eval()
    try:
        gen.load_state_dict(torch.load(_gan_weights_path(), map_location=device()))
        _gan_gen = gen
    except Exception as e:
        print(f"[GAN] Could not load weights: {e}")
        _gan_gen = gen  # return untrained generator
    return _gan_gen

# ---------------- Request Models ----------------
class GanSampleReq(BaseModel):
    n: int = 16
    z_dim: int = 100

class DiffusionGenerateReq(BaseModel):
    n_images: int = 16
    img_size: int = 32
    img_channels: int = 3
    steps: int = 1000
    train_if_missing: bool = False
    train_epochs: int = 1
    max_batches: Optional[int] = 50

class EBMGenerateReq(BaseModel):
    steps: int = 200
    step_size: float = 0.05
    img_size: int = 32
    img_channels: int = 3
    noise_scale: float = 0.01

class TrainDiffusionReq(BaseModel):
    epochs: int = 1
    steps: int = 1000
    batch_size: int = 128
    lr: float = 1e-3
    max_batches: Optional[int] = 100
    img_channels: int = 3

class TrainEBMReq(BaseModel):
    epochs: int = 1
    batch_size: int = 64
    lr: float = 1e-4
    img_channels: int = 3
    max_batches: Optional[int] = 100

@app.get("/")
def root():
    return {"service": "image-generation", "device": device()}

# ---------------- Diffusion helpers ----------------
def _load_diffusion(req: DiffusionGenerateReq):
    global _diffusion_model
    if _diffusion_model is None:
        weights = DIFFUSION_WEIGHTS if DIFFUSION_WEIGHTS.exists() else None
        _diffusion_model = load_or_init_diffusion(str(weights) if weights else None, device=device(), img_channels=req.img_channels)
        if weights is None and req.train_if_missing:
            ds = datasets.CIFAR10(root=_repo_root / "data", train=True, download=True, transform=transforms.ToTensor())
            dl = DataLoader(ds, batch_size=128, shuffle=True)
            train_diffusion(_diffusion_model, dl, device=device(), epochs=req.train_epochs, T=req.steps, lr=1e-3, max_batches=req.max_batches)
            torch.save(_diffusion_model.state_dict(), DIFFUSION_WEIGHTS)
            print("[Diffusion] Auto-trained and saved weights.")
    return _diffusion_model

# ---------------- EBM helpers ----------------
def _load_ebm(channels: int):
    global _ebm_model
    if _ebm_model is None:
        _ebm_model = EnergyModel(in_channels=channels).to(device())
        if EBM_WEIGHTS.exists():
            try:
                _ebm_model.load_state_dict(torch.load(EBM_WEIGHTS, map_location=device()))
                print("[EBM] Loaded weights.")
            except Exception as e:
                print(f"[EBM] Load failed: {e}")
    return _ebm_model

# ---------------- Endpoints ----------------
@app.post("/gan/sample", response_class=Response)
def gan_sample(req: GanSampleReq):
    gen = _load_gan(req.z_dim)
    with torch.no_grad():
        z = torch.randn(req.n, req.z_dim, device=device())
        imgs = gen(z).cpu()
        imgs = (imgs + 1)/2  # [-1,1] -> [0,1]
        grid = make_grid(imgs, nrow=int(math.ceil(req.n**0.5)), padding=2)
        buf = BytesIO(); save_image(grid, buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/generate/diffusion")
def generate_diffusion(req: DiffusionGenerateReq):
    model = _load_diffusion(req)
    imgs = sample_ddpm(model, num_samples=req.n_images, img_size=req.img_size, img_channels=req.img_channels, T=req.steps, device=device())
    imgs = imgs.clamp(0,1)
    result = []
    for arr in imgs.permute(0,2,3,1).cpu().numpy():
        pil = Image.fromarray((arr*255).round().astype('uint8'))
        b = BytesIO(); pil.save(b, format='PNG')
        result.append(base64.b64encode(b.getvalue()).decode())
    return {"images": result, "count": len(result)}

@app.post("/generate/diffusion/png", response_class=Response)
def generate_diffusion_png(req: DiffusionGenerateReq):
    model = _load_diffusion(req)
    imgs = sample_ddpm(model, num_samples=req.n_images, img_size=req.img_size, img_channels=req.img_channels, T=req.steps, device=device())
    imgs = imgs.clamp(0,1)
    grid = make_grid(imgs, nrow=int(math.ceil(req.n_images**0.5)))
    buf = BytesIO(); save_image(grid, buf, format="PNG")
    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/train/diffusion")
def train_diffusion_endpoint(req: TrainDiffusionReq):
    gen_req = DiffusionGenerateReq(train_if_missing=True, train_epochs=req.epochs, steps=req.steps, img_channels=req.img_channels)
    model = _load_diffusion(gen_req)
    ds = datasets.CIFAR10(root=_repo_root / "data", train=True, download=True, transform=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=req.batch_size, shuffle=True)
    train_diffusion(model, dl, device=device(), epochs=req.epochs, T=req.steps, lr=req.lr, max_batches=req.max_batches)
    torch.save(model.state_dict(), DIFFUSION_WEIGHTS)
    return {"status": "ok", "weights": str(DIFFUSION_WEIGHTS)}

@app.post("/generate/ebm")
def generate_ebm(req: EBMGenerateReq):
    model = _load_ebm(req.img_channels)
    x, _ = sample_ebm(model, steps=req.steps, lr=req.step_size, noise_scale=req.noise_scale, device=device(), size=req.img_size, channels=req.img_channels, batch_size=1)
    arr = (x[0].permute(1,2,0).detach().numpy()*255).clip(0,255).astype('uint8')
    pil = Image.fromarray(arr); buf = BytesIO(); pil.save(buf, format='PNG')
    return {"image": base64.b64encode(buf.getvalue()).decode(), "shape": arr.shape}

@app.post("/generate/ebm/png", response_class=Response)
def generate_ebm_png(req: EBMGenerateReq):
    model = _load_ebm(req.img_channels)
    x, _ = sample_ebm(model, steps=req.steps, lr=req.step_size, noise_scale=req.noise_scale, device=device(), size=req.img_size, channels=req.img_channels, batch_size=1)
    arr = (x[0].permute(1,2,0).detach().numpy()*255).clip(0,255).astype('uint8')
    pil = Image.fromarray(arr); buf = BytesIO(); pil.save(buf, format='PNG')
    return Response(content=buf.getvalue(), media_type="image/png")

@app.post("/train/ebm")
def train_ebm_endpoint(req: TrainEBMReq):
    model = _load_ebm(req.img_channels)
    ds = datasets.CIFAR10(root=_repo_root / "data", train=True, download=True, transform=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=req.batch_size, shuffle=True)
    train_ebm(model, dl, device=device(), epochs=req.epochs, lr=req.lr, max_batches=req.max_batches)
    torch.save(model.state_dict(), EBM_WEIGHTS)
    return {"status": "ok", "weights": str(EBM_WEIGHTS)}
