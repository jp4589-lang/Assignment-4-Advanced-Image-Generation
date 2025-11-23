import torch, argparse
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from pathlib import Path
from app.ebm import EnergyModel, train_ebm

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=3)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--max_batches", type=int, default=30)
    a = p.parse_args()
    dev = ("cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu"))
    root = Path(__file__).resolve().parent
    data_dir = root / "data"; weights = root / "models" / "ebm_cifar10.pt"; weights.parent.mkdir(parents=True, exist_ok=True)
    ds = datasets.CIFAR10(root=str(data_dir), train=True, download=True, transform=transforms.ToTensor())
    dl = DataLoader(ds, batch_size=a.batch_size, shuffle=True)
    model = EnergyModel(in_channels=3).to(dev)
    train_ebm(model, dl, device=dev, epochs=a.epochs, lr=a.lr, max_batches=a.max_batches)
    torch.save(model.state_dict(), weights)
    print(f"Saved EBM weights to {weights}")
