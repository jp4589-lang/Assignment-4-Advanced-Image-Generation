import torch, os, random, numpy as np
from torch import nn

def set_seed(seed: int = 42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True; torch.backends.cudnn.benchmark = False

def get_device():
    return ("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))

def save_model(model: nn.Module, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True); torch.save(model.state_dict(), path); print(f"Saved model: {path}")

def load_model(model: nn.Module, path: str, map_location=None):
    model.load_state_dict(torch.load(path, map_location=map_location)); print(f"Loaded weights: {path}")

def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
