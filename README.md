# NanoClear: AI-Based Image Restoration & Super-Resolution for Semiconductor Inspection

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![CUDA](https://img.shields.io/badge/CUDA-Enabled-green)

**NanoClear** is an end-to-end deep learning pipeline designed for simultaneous **speckle despeckling, Gaussian deblurring, and 2x super-resolution** on single-channel grayscale semiconductor wafer inspection images.

Built for the **KLA Semiconductor Image Restoration Challenge**, NanoClear utilizes a modified **NAFNet (Nonlinear Activation-Free Network)** architecture with **PixelShuffle (Sub-Pixel Convolution)** to restore fine grating lines and trace structures while optimizing for ultra-fast GPU inference.

---

## 🔬 Core Architecture & Methodology

1. **Intensity Normalization**: Adaptive dynamic percentile clipping ($[p_{0.1}, p_{99.9}]$) to handle out-of-range speckle noise intensity spikes cleanly.
2. **Feature Backbone (NAFBlock)**: Uses **Simple Gate** mechanisms and Depthwise Convolutions instead of heavy Self-Attention to maintain high structural fidelity at high execution speeds.
3. **Super-Resolution Head**: $2\times$ upsampling powered by Sub-Pixel Convolution (`PixelShuffle`), eliminating checkerboard artifacts on low-resolution inputs ($128\times128 \to 256\times256$ or $256\times256 \to 512\times512$).
4. **Composite Hybrid Loss**: Combined **Charbonnier Loss** ($L_1$ variant) + **Sobel Edge Gradient Loss** to force sharp boundary reconstruction on blurred wafer traces.

---

## 📁 Repository Structure

```text
nanoclear/
├── models/
│   └── nafnet_sr.py       # NAFNetSR model architecture definition
├── dataset.py            # NumPy (.npy) dataset loader & dynamic normalization
├── train.py              # End-to-end model training engine
├── evaluate.py           # Standalone benchmark inference script (for KLA evaluation)
├── weights/
│   └── model_final.pt    # Pre-trained model weights
├── requirements.txt      # Dependency environment specs
├── .gitignore            # Excludes datasets & pycache
└── README.md             # Documentation# Nanoclear