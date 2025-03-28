# HufuNetPr: Watermarking and Attack Simulation for Deep Neural Networks

Deep Neural Networks (DNNs) have achieved state-of-the-art performance across various domains. However, they are vulnerable to intellectual property (IP) infringement via unauthorized use or replication.

This project simulates the **HufuNet** watermarking framework as proposed by Lv et al. in:

> **[A Robustness-Assured White-Box Watermark in Neural Networks](https://ieeexplore.ieee.org/abstract/document/10038500)**  

---

## Table of Contents
- [Overview](#overview)
- [Features](#features)
- [Requirements](#requirements)
- [Getting Started](#getting-started)
- [Running the Code](#running-the-code)
- [Configuration](#configuration)
- [Example Usage](#example-usage)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [License](#license)

---

## Overview

HufuNet is trained iteratively alongside the target model to be watermarked and the autoencoder. The autoencoder consists of two main components:

- The encoder embeds watermark signals into the model’s internal weights.
- The decoder is used for ownership verification post-deployment.

Attack simulations evaluate the watermark’s robustness against various adversarial strategies.

---

## Features

- Watermarked model training with dynamic encoder-based watermark embedding  
- Non-watermarked baseline model training for comparison  
- Autoencoder training:
  - HufuNet (benign encoder)
  - AttackerAE (malicious encoder simulating overwriting attacks)  
- Attack simulation modules:
  - Pruning attack
  - Fine-tuning attack
  - Watermark overwriting attack  
- White-box extraction and verification of embedded watermarks

---

## Requirements

- Python >= **3.10**

### Python Dependencies

```bash
torch ~= 2.6.0
torchvision ~= 0.21.0
numpy ~= 2.2.3
matplotlib ~= 3.10.1
setuptools ~= 76.0.0
tqdm ~= 4.67.1
```

---

## Getting Started

### 1. Clone the Repository

```bash
git clone https://github.com/Tura777/HufuNetSimulation-.git
cd HufuNetSimulation-
```

### 2. Create and activate a Conda environment

```bash
conda create -n hufunet python=3.12
conda activate hufunet
```

### 3. Install the required dependencies

```bash
pip install -r requirements.txt
```

---

## Running the Code  

You can either run individual scripts or use the unified `run.py` launcher.

### Option A: Run Scripts Directly

```bash
# Train a watermarked model
python trainWatermarkModels/trainMainModel.py --model_name CNN --total_rounds 1

# Simulate watermark attacks only
python watermarkTrainer.py --model_name CNN --attack_only

# Train a non-watermarked baseline model
python trainNonWatermarkModel/trainNonWMMainModel.py --model_name CNN --total_rounds 1

# Train a dummy baseline model
python trainNonWatermarkModel/trainDummyModel.py --model_name CNN --total_rounds 1

# Train autoencoders
python trainAutoEncoders/autoencoder_trainer.py --model_name HufuNet --total_rounds 1
python trainAutoEncoders/autoencoder_trainer.py --model_name AttackerAE --total_rounds 1
```

### Option B: Use the Unified Launcher

```bash
# Train watermarked model
python run.py watermark --model_name CNN --total_rounds 1

# Simulate attacks on watermarked model only
python run.py watermark --model_name MLP_RIGA --attacks_only

# Train non-watermarked model
python run.py nonwatermark --model_name CNN --total_rounds 1

# Train autoencoder
python run.py autoencoder --model_name HufuNet --total_rounds 1

# Train dummy model
python run.py dummy --model_name MLP --total_rounds 1
```

---

## Configuration

Configuration options such as model architecture, training parameters, watermark type, and attack settings are defined in the `config.py` file. You can modify these settings to customize the training process.

---

## Example Usage

Here are some example use cases to get you started:

1. **Training a watermarked model**:
   ```bash
   python run.py watermark --model_name CNN --total_rounds 1
   ```

2. **Simulating watermark attacks**:
   ```bash
   python run.py watermark --model_name MLP_RIGA --attacks_only
   ```

3. **Training a non-watermarked baseline model**:
   ```bash
   python run.py nonwatermark --model_name CNN --total_rounds 1
   ```

4. **Training a dummy model**:
   ```bash
   python run.py dummy --model_name MLP --total_rounds 1
   ```

---

## Project Structure

```
HufuNetSimulation/
│
├── configs/
│      └── config.py                  # Configuration settings
├── models/
|     └── cnn.py                      # CNN model architecture
|     └── mlp.py                      # MLP model architecture
|     └── etc.
│
├── trainAutoEncoders/
│   └── autoencoder_trainer.py        # Train HufuNet and AttackerAE
│
├── trainNonWatermarkModel/
│   ├── trainNonWMMainModel.py        # Train non-watermarked baseline models
│   └── trainDummyModel.py            # Train dummy model (no watermark)
│
├── trainWatermarkModels/
│   └── trainMainModel.py             # Train watermarked models
│
├── run.py                            # Entry point
├── requirements.txt                  # Dependency list
└── README.md                         # Project documentation (this file)
```

---

## Contributing

Contributions are welcome! Please follow these steps to contribute:

1. Fork the repository.
2. Create a new branch (`git checkout -b feature-branch`).
3. Make your changes.
4. Commit your changes (`git commit -am 'Add new feature'`).
5. Push to the branch (`git push origin feature-branch`).
6. Create a new Pull Request.

---

## License

This project is licensed under the MIT License.