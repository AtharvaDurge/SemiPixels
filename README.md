# NanoClear: Deep Learning Framework for Semiconductor Image Restoration & 2× Super-Resolution

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**NanoClear** is an end-to-end deep learning framework designed for simultaneous **speckle despeckling, Gaussian deblurring, and 2× super-resolution** on single-channel grayscale semiconductor wafer inspection imagery. 

Developed for high-precision metrology challenges, NanoClear leverages a high-capacity **Nonlinear Activation-Free Network (NAFNet-SR)** architecture coupled with **Sub-Pixel Convolution (PixelShuffle)** and **8-fold Test-Time Augmentation (TTA)** to restore fine grating lines, sub-micron circuit traces, and critical structural edges.

---

## 🔬 Core Methodology & Architecture

* **Nonlinear Activation-Free Blocks (NAFBlock)**: Replaces computationally expensive self-attention and non-linear activations (like ReLU/GELU) with **Simple Gates** and depthwise separable convolutions to maximize throughput and gradient flow.
* **Sub-Pixel Super-Resolution Head**: Employs $2\times$ PixelShuffle upsampling ($256\times256 \to 512\times512$), preventing checkerboard deconvolution artifacts across high-frequency nano-scale patterns.
* **Multi-Scale Loss Supervision**: Utilizes a dual-phase training scheme:
  * **Phase 1**: Hybrid Charbonnier Loss + Structural SSIM Loss for macro-structure convergence.
  * **Phase 2 (Fine-tuning)**: High-ratio Charbonnier residual optimization with Cosine Annealing learning rate schedules to minimize pixel-level Mean Squared Error ($\text{MSE}$).
* **8-Fold Test-Time Augmentation (TTA)**: Evaluates input batches across all 4 spatial rotations and horizontal flips, averaging predictions to suppress stochastic noise and enhance line continuity.

---

## 📁 Repository Structure

```text
Nanoclear/
├── models/
│   └── nafnet_sr.py       # High-capacity NAFNetSR architecture (width=48)
├── data/
│   ├── train/
│   │   ├── degraded/      # Low-resolution noisy input images
│   │   └── gt/            # High-resolution ground truth reference images
│   └── test/              # Evaluation test split
├── weights/
│   └── model_best.pt      # Optimal saved checkpoint
├── dataset.py             # Image pair loader, augmentations & patch cropping
├── losses.py              # Charbonnier, SSIM, and Edge Gradient loss modules
├── train.py               # Main Stage-1 training script
├── finetune.py            # Stage-2 PSNR fine-tuning engine
├── check_score.py         # Validation benchmark script (PSNR / SSIM with TTA)
├── evaluate.py            # Batch inference script for test sets
├── requirements.txt       # Python environment dependencies
├── .gitignore             # Git exclusion rules
└── README.md              # Project documentation