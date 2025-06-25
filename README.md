# Z2N2N: Zero-Shot Denoising Using Noise2Noise

Z2N2N is a deep learning framework for image denoising that requires only a single noisy input image. The system automatically generates similar noisy samples, trains a self-supervised model, and outputs a denoised version of the input.

This repository includes both the model and a minimal user interface for testing and experimentation.

---

## Features

- One-image input: provide a single noisy image.
- Self-supervised training based on Noise2Noise principles.
- Automatic generation of synthetic noisy pairs.
- Simple web UI to upload images and view results.
- Lightweight and easy to run locally.

---

## Demo

![UI Demo](assets/ui-demo.png)

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/z2n2n.git
cd z2n2n
pip install -r requirements.txt
python app.py
