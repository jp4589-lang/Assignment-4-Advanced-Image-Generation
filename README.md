# Assignment 4: Advanced Image Generation (Deployment Focus)

Minimal FastAPI + Docker service exposing:
- Diffusion (DDPM-lite) CIFAR-10 sampler
- Energy-Based Model (EBM) CIFAR-10 sampler (Langevin dynamics)
- Optional MNIST GAN sample endpoint (if `outputs/gan_mnist.pt` is mounted)

> Limited training (2–3 epochs with batch caps) was intentional to prioritize showing a working, reproducible deployment. Full training for clean samples would take 10+ hours of compute.

## Endpoints
| Method | Path | Purpose |
|--------|------|---------|
| GET | / | Health + device info |
| POST | /generate/diffusion | Return base64 PNG samples (list) |
| POST | /generate/diffusion/png | Return a PNG grid |
| POST | /generate/ebm | Return single base64 PNG from EBM |
| POST | /generate/ebm/png | Return single PNG from EBM |
| POST | /train/diffusion | (Optional) quick diffusion training |
| POST | /train/ebm | (Optional) quick EBM training |
| POST | /gan/sample | MNIST GAN grid (requires weight) |

## Quick Start (Docker)
```bash
# From repo root
docker build -t image-gen-api .
docker run --rm -p 8000:8000 \
  -v "$(pwd)/outputs:/app/outputs" \
  -v "$(pwd)/models:/app/models" \
  image-gen-api
# Swagger UI:
# http://localhost:8000/docs
```

## Pretrained Weights
The repo includes pretrained CIFAR-10 checkpoints under `models/`:
- `models/diffusion_cifar10.pt`
- `models/ebm_cifar10.pt`

The API auto-loads these at startup, so you can run Docker and immediately use the `/generate/*` endpoints—no training required. The training endpoints remain optional for experimentation.

## Local Dev
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

## Sample Calls
```bash
# Diffusion grid (PNG)
curl -X POST http://localhost:8000/generate/diffusion/png \
  -H 'Content-Type: application/json' \
  -d '{"n_images":16,"steps":200}' \
  -o diffusion_grid.png

# EBM single image
curl -X POST http://localhost:8000/generate/ebm/png \
  -H 'Content-Type: application/json' \
  -d '{"steps":150}' \
  -o ebm_sample.png
```

## Training (Under-Trained Demonstration)
```bash
python train_diffusion_cifar10.py --epochs 2 --max_batches 50
python train_ebm_cifar10.py --epochs 3 --max_batches 30
```

## Why Images Look Noisy / Single-Color
- Diffusion: Only a fraction of CIFAR-10 seen → early denoising skills show coarse shapes & colors.
- EBM: Mode collapse toward a simple energy minimum → flat or single-color patterns.

Improvement path (not required here): more epochs, remove batch caps, better normalization, EMA for diffusion, tune Langevin `steps/step_size/noise_scale`.

## Included Files
```
Dockerfile
requirements.txt
app/main.py
app/diffusion.py
app/ebm.py
helper_lib/diffusion_model.py
helper_lib/model.py
helper_lib/utils.py
train_diffusion_cifar10.py
train_ebm_cifar10.py
README.md
```

## License
Academic / coursework use only.

---
Deployed focus complete. Ready for evaluation.
