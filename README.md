---
title: SparsePlug - Variable Compression Platform
emoji: 🔥
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
license: cc-by-nc-4.0
short_description: One model. Any device. Zero tweaks. Production-grade sparse inference.
---

# SparsePlug™ - Variable Compression Platform

**One model. Any device. Zero tweaks.**

[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](https://creativecommons.org/licenses/by-nc/4.0/)
[![Production Ready](https://img.shields.io/badge/status-production-green.svg)]()

SparsePlug is the world's first **Variable Compression Platform** powered by the **UPG-PAC V2** engine. Deploy Large Language Models at extreme sparsity levels (50-99%) while preserving semantic coherence through **Geometric Intelligence**.

> **⚠️ PROPRIETARY TECHNOLOGY**: This platform contains patented algorithms. See LICENSE for usage terms.

## 🚀 Features

- ✅ **Variable Sparsity**: 50-99% compression at runtime
- ✅ **5.58x Faster**: Rust-accelerated inference
- ✅ **96% Compression**: 1B params in <200MB
- ✅ **Edge-Ready**: Raspberry Pi, mobile, IoT
- ✅ **Production API**: RESTful with streaming

## 📦 Quick Start

```bash
docker run -p 8000:8000 ghcr.io/koba42coo/sparseplug:latest
curl http://localhost:8000/health
```

See [PRODUCTION_DEPLOY.md](PRODUCTION_DEPLOY.md) for full deployment guide.

## ⚖️ License

CC BY-NC 4.0 - Non-commercial use only. Contact licensing@sparseplug.ai for commercial licensing.

**SparsePlug™** and **UPG-PAC V2** are proprietary technologies. Unauthorized commercial use or reverse engineering is prohibited.
