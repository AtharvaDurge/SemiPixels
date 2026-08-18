# NanoClear: Deep Learning Framework for Semiconductor Image Restoration & 2× Super-Resolution

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green)

**NanoClear** is an end-to-end deep learning framework designed for simultaneous speckle despeckling, Gaussian deblurring, and 2× super-resolution on single-channel grayscale semiconductor wafer inspection imagery[cite: 5].

Developed for high-precision metrology challenges, NanoClear leverages a **Nonlinear Activation-Free Network (NAFNet-SR)** architecture coupled with **Sub-Pixel Convolution (PixelShuffle)** and **8-fold Test-Time Augmentation (TTA)** to restore fine grating lines, sub-micron circuit traces, and critical structural edges[cite: 1, 5].

---

## 📁 Repository Structure

```text
Nanoclear/
├── models/
│   └── nafnet_sr.py       # NAFNetSR model architecture definition[cite: 5, 8]
├── weights/
│   └── model_best.pt      # Pre-trained checkpoint weights[cite: 5]
├── run.py                 # Standalone inference entrypoint[cite: 1, 5]
├── requirements.txt       # Python environment dependencies[cite: 6]
└── README.md              # Project documentation[cite: 5]