
# NanoClear: Deep Learning Framework for Semiconductor Image Restoration & 2× Super-Resolution

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-orange)
![CUDA](https://img.shields.io/badge/CUDA-Accelerated-green)

**NanoClear** is an end-to-end deep learning framework designed for simultaneous speckle despeckling, Gaussian deblurring, and 2× super-resolution on single-channel grayscale semiconductor wafer inspection imagery.

Developed for high-precision metrology challenges, NanoClear leverages a **Nonlinear Activation-Free Network (NAFNet-SR)** architecture coupled with **Sub-Pixel Convolution (PixelShuffle)** and **8-fold Test-Time Augmentation (TTA)** to restore fine grating lines, sub-micron circuit traces, and critical structural edges[cite: 1].

---

## 📁 Repository Structure

```text
Nanoclear/
├── models/
│   └── nafnet_sr.py       # NAFNetSR architecture definition
├── weights/
│   └── model_best.pt      # Pre-trained checkpoint weights
├── run.py                 # Standalone inference entrypoint[cite: 1]
├── requirements.txt       # Python environment dependencies
└── README.md              # Project documentation

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

### Pipeline Features & Guarantees

* **Flexible Discovery**: Recursively searches for `.npy` arrays whether stored directly in `<input-dir>` or inside nested subdirectories.


* **Automated Directory Handling**: Creates `<output-dir>` automatically if it does not already exist.


* **Strict Format Compliance**: Saves one restored 2D `.npy` array per input file with matching filenames.


* **Bounded Output Range**: Output pixel values are strictly normalized and clamped inside `[0.0, 1.0]` with zero `NaN` or `Inf` artifacts.


* **Self-Contained & Offline**: Runs fully offline using local GPU acceleration without external API calls or network connectivity.



---

## 🛠️ Tech Stack & Requirements

* **Language**: Python 3.10+
* **Deep Learning**: PyTorch (CUDA-enabled), Torchvision
* **Numerical & Image Processing**: NumPy, OpenCV, Scikit-Image, Pillow
