# Installing PyTorch — Platform-Specific

PyTorch must be installed **before** `pip install -r requirements.txt` because the
correct wheel varies by platform.

## macOS (Apple Silicon / MPS)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```
> MPS backend is included in the CPU wheel on macOS 12.3+. No special build needed.

## Ubuntu — CUDA 11.8 (GTX 1080 / RTX 3070)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

## Ubuntu — CUDA 12.1
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

## CPU only (any platform)
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

---

After PyTorch is installed, install the rest:
```bash
pip install -r requirements.txt
```
