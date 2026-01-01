# Contributing to SparsePlug

Thank you for your interest in SparsePlug! We are building the future of efficient, edge-ready AI, and we need your help.

## How to Contribute

### 1. Report Bugs
Find a crash or a weird UI glitch? Open an Issue on GitHub with:
- Steps to reproduce
- Model file used (size/type)
- Browser/OS info

### 2. Feature Requests
Want a new feature? (e.g., "Support for .gguf files", "Visual heatmap of sparsity"). Open a Discussion or Issue!

### 3. Edge Testing (High Priority 🚨)
We want SparsePlug to run on *everything*.
If you have:
- **Raspberry Pi 4/5**
- **NVIDIA Jetson**
- **Apple M1/M2/M3**
- **Old Intel Laptops**

Please try running the Docker container locally and report your performance benchmarks!
- Build time
- Inference speed (if applicable)
- RAM usage

### 4. Code Contributions
1. Fork the repo
2. Create a branch (`git checkout -b feature/cool-new-thing`)
3. Commit changes (`git commit -m 'Added cool thing'`)
4. Push to branch (`git push origin feature/cool-new-thing`)
5. Open a Pull Request

## Development Setup

```bash
# Install deps
pip install -r requirements.txt

# Compile Rust Kernel (Requires Cargo)
cd core/upg_kernel
maturin develop --release

# Run Local Server
python demo/app.py
```

## License
By contributing, you agree that your contributions will be licensed under the project's [LICENSE](LICENSE).
