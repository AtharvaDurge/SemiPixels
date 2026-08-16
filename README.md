
# NanoClear: Deep Learning Framework for Semiconductor Image Restoration & 2× Super-Resolution

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green)
![License](https://img.shields.io/badge/License-MIT-lightgrey)

**NanoClear** is an end-to-end deep learning framework designed for simultaneous **speckle despeckling, Gaussian deblurring, and 2× super-resolution** on single-channel grayscale semiconductor wafer inspection imagery[cite: 6].

Developed for high-precision metrology challenges, NanoClear leverages a **Nonlinear Activation-Free Network (NAFNet-SR)** architecture coupled with **Sub-Pixel Convolution (PixelShuffle)** and **8-fold Test-Time Augmentation (TTA)** to restore fine grating lines, sub-micron circuit traces, and critical structural edges[cite: 1, 6].

---

## 🔬 Core Methodology & Architecture

* **Nonlinear Activation-Free Blocks (NAFBlock)**: Replaces computationally expensive self-attention and traditional non-linear activations (such as ReLU/GELU) with **Simple Gates** and depthwise separable convolutions to maximize throughput and gradient flow[cite: 6].
* **Sub-Pixel Super-Resolution Head**: Employs $2\times$ PixelShuffle upsampling ($128\times128 \to 256\times256$ or $256\times256 \to 512\times512$), preventing checkerboard deconvolution artifacts across high-frequency nano-scale patterns[cite: 6].
* **Multi-Scale Loss Supervision**: Utilizes a structured two-stage training scheme:
  * **Stage 1 (Base Training)**: Fast Composite Loss combining **Charbonnier Loss**, **Sobel Edge Gradient Loss**, and **Fast SSIM Loss** for macro-structure convergence[cite: 5].
  * **Stage 2 (Fine-Tuning)**: Direct Charbonnier residual pixel loss with Cosine Annealing learning rate schedules to minimize pixel-level Mean Squared Error ($\text{MSE}$) and maximize PSNR[cite: 4].
* **8-Fold Test-Time Augmentation (TTA)**: Evaluates input batches across all 4 spatial rotations and horizontal flips, averaging predictions to suppress stochastic noise and enhance line continuity[cite: 1, 3].

---

## 📁 Repository Structure

```text
Nanoclear/
├── models/
│   └── nafnet_sr.py       # NAFNetSR model architecture definition
├── data/                  # Local dataset directory (not tracked in Git)
│   ├── train/
│   │   ├── degraded/      # Low-resolution noisy input images (.npy)
│   │   └── gt/            # High-resolution ground truth reference images (.npy)
│   └── test/              # Evaluation test images (.npy)
├── weights/
│   └── model_best.pt      # Optimal saved checkpoint
├── dataset.py             # NumPy (.npy) dataset loader, augmentations & patch cropping
├── losses.py              # Loss function definitions
├── train.py               # Main Stage-1 composite loss training script
├── finetune.py            # Stage-2 PSNR fine-tuning engine
├── check_score.py         # Validation benchmark script (PSNR / SSIM with 8-fold TTA)
├── evaluate.py            # Standalone batch inference script with CLI arguments & TTA
├── requirements.txt       # Python environment dependencies
├── .gitignore             # Git exclusion rules
└── README.md              # Project documentation

```

---

## ⚙️ Installation & Setup

### 1. Clone the Repository

```bash
git clone [https://github.com/](https://github.com/)<your-username>/Nanoclear.git
cd Nanoclear

```

### 2. Set Up Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

```

### 3. Install Dependencies

```bash
pip install -r requirements.txt

```

### 4. Dataset Setup

Place your dataset files (`.npy` or image arrays) inside the `data/` directory matching the following layout:

```text
data/
├── train/
│   ├── degraded/   # Input low-resolution/noisy files
│   └── gt/         # Ground truth target reference files
└── test/           # Test files for inference

```

---

## 🚀 How to Run

### Step 1: Base Training (Stage 1)

Train the NAFNet-SR model using the composite Charbonnier + Sobel Edge + SSIM loss:

```bash
python train.py

```

Saves checkpoints to `weights/model_best.pt` and `weights/model_final.pt`.

### Step 2: Fine-Tuning for PSNR (Stage 2)

Resume from the base checkpoint and fine-tune using direct pixel loss with cosine learning rate decay:

```bash
python finetune.py

```

### Step 3: Run In-Distribution Benchmark Check

Evaluate validation performance (PSNR and SSIM) using 8-fold Test-Time Augmentation on your local data:

```bash
python check_score.py

```

### Step 4: Run Batch Test Inference

Generate restored super-resolved outputs (`.npy` arrays and `.png` visualizations) for unlabeled test data:

```bash
# Standard inference
python evaluate.py --input_dir data/test --output_dir output/

# High-precision inference with 8-fold TTA enabled
python evaluate.py --input_dir data/test --output_dir output/ --tta

# Custom weights path
python evaluate.py --input_dir data/test --output_dir output/ --weights weights/model_best.pt --tta

```

---

## 📊 Performance Benchmarks

| Model Configuration | Scale Factor | PSNR (dB) | SSIM | 8-Fold TTA |
| --- | --- | --- | --- | --- |
| Bicubic Interpolation | $2\times$ | 21.40 dB | 0.5820 | No |
| NAFNet-SR (Stage 1 Base) | $2\times$ | 23.12 dB | 0.6940 | No |
| **NAFNet-SR (Stage 2 PSNR Boost)** | $2\times$ | **24.00+ dB** | **0.7200+** | **Yes** |

---

## 🛠️ Tech Stack & Requirements

* **Language**: Python 3.10+
* **Deep Learning**: PyTorch (CUDA-enabled), Torchvision
* **Image Processing & Metrology**: NumPy, OpenCV, Scikit-Image, Pillow
* **Progress Tracking**: TQDM



