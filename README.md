
# NanoClear: Deep Learning Framework for Semiconductor Image Restoration & 2× Super-Resolution

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green)

**NanoClear** is an end-to-end deep learning framework designed for simultaneous speckle despeckling, Gaussian deblurring, and 2× super-resolution on single-channel grayscale semiconductor wafer inspection imagery.

Developed for high-precision metrology challenges, NanoClear leverages a **Nonlinear Activation-Free Network (NAFNet-SR)** architecture coupled with **Sub-Pixel Convolution (PixelShuffle)** and **8-fold Test-Time Augmentation (TTA)** to restore fine grating lines, sub-micron circuit traces, and critical structural edges.

---

## 📁 Repository Structure

```text
Nanoclear/
├── models/
│   └── nafnet_sr.py        # NAFNetSR architecture definition
├── weights/
│   └── model_best.pt       # Pre-trained checkpoint weights
├── run.py                  # Standalone inference entrypoint
├── requirements.txt        # Python environment dependencies
└── README.md               # Project documentation

```

---

## ⚙️ Installation & Setup

### 1. Set Up Virtual Environment

```bash
python -m venv venv
# On Windows:
venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

```

### 2. Install Dependencies

```bash
pip install -r requirements.txt

```

---

## 🚀 How to Run

Execute standalone batch inference by passing the input and output directories as positional arguments:

```bash
python run.py <input-dir> <output-dir>

```

### Example

```bash
python run.py data/test output

```

---

## 🧠 Model Architecture

NanoClear uses a **NAFNet-SR** based restoration architecture consisting of:

1. **Input Processing** – Accepts single-channel grayscale `.npy` arrays.
2. **NAFNet Feature Extraction** – Extracts and restores spatial features using nonlinear activation-free blocks with simple gate and channel attention mechanisms.
3. **Image Restoration** – Performs joint despeckling, denoising, and deblurring.
4. **PixelShuffle Upsampling** – Performs 2× spatial super-resolution using sub-pixel convolution.
5. **Test-Time Augmentation (TTA)** – Evaluates 8 geometric variations ($4\text{ rotations} \times 2\text{ flips}$) during inference and averages predictions for enhanced structural fidelity.
6. **Output Normalization** – Produces 2D floating-point arrays strictly bounded to `[0.0, 1.0]`.

```text
Input Image (.npy)
     │
     ▼
NAFNet-SR Encoder-Decoder
     │
     ├── Feature Extraction
     ├── Despeckling
     └── Deblurring
     │
     ▼
PixelShuffle (2× Upscaling)
     │
     ▼
8-Fold Test-Time Augmentation (TTA)
     │
     ▼
Prediction Aggregation & Normalization
     │
     ▼
Restored 2× Image
```



---

## 📋 Pipeline Features & Guarantees

* **Flexible Discovery**: Recursively searches for `.npy` arrays whether stored directly in `<input-dir>` or inside nested subdirectories.
* **Automated Directory Handling**: Automatically creates `<output-dir>` if it does not already exist.
* **Strict Format Compliance**: Saves one restored 2D `.npy` array per input file with matching filenames.
* **Bounded Output Range**: Output pixel values are strictly normalized and clamped inside `[0.0, 1.0]` with zero `NaN` or `Inf` artifacts.
* **Self-Contained & Offline**: Operates fully offline using local GPU acceleration without external API calls or network connectivity.

---

## 🛠️ Tech Stack & Requirements

* **Language**: Python 3.10+
* **Deep Learning**: PyTorch (CUDA-enabled), Torchvision
* **Numerical & Image Processing**: NumPy, OpenCV, Scikit-Image, Pillow

---

## 👥 Team Members

* **Atharva Durge** – [@AtharvaDurge](https://github.com/AtharvaDurge)
* **Aryan Parab** – [@ap6458](https://github.com/ap6458)

