# OOD-Detection

This repository contains the official implementation of the paper:

**"YouTube Video Identification with Out-of-Distribution (OOD) Detection in Encrypted Network Traffic"**  
_Muhammad B. Sarwar, Syed M. Ahmad, Maheera Amjad, Muhammad U. S. Khan_

---

## 📘 Overview

This work proposes a deep learning framework for real-time YouTube video identification from encrypted network traffic, with integrated out-of-distribution (OOD) detection. The framework processes packet metadata (not payloads) to preserve privacy and utilizes hybrid CNN-LSTM architectures combined with custom loss functions and adaptive thresholding.

---

## 📌 Key Features

- 📦 **Encrypted Traffic Classification**: No decryption needed—classification is based on metadata like BPS vectors, packet sizes, and inter-arrival timings.
- 🧠 **Deep Learning Models**: Implements SCNN, SCNN-LSTM, AlexNet, LeNet, and BPS variants.
- 🎯 **OOD Detection**: Uses percentile- and ROC-based thresholding to flag unknown video classes.
- ⚙️ **Loss Function Engineering**: Models are trained with combinations of Cross-Entropy and KL Divergence to improve generalization.
- 📊 **Daily Evaluation**: Models validated over a 14-day dataset of 30 known and 13 unseen YouTube classes.

---

## 🧪 Models

| Model         | Description                                                      |
|---------------|------------------------------------------------------------------|
| SCNN          | Sequential CNN with 3 conv layers and dropout                    |
| SCNN-LSTM     | SCNN + LSTM for temporal modeling of traffic patterns            |
| AlexNet       | Adapted for 1D convolutions over BPS vectors                     |
| LeNet         | Simpler convnet baseline                                         |
| BPS           | Byte-per-second histogram-based model                            |

Each model is trained with:
- CrossEntropy + KL Divergence
- CrossEntropy / KL Divergence
- Combined custom loss

---

## 📁 Project Structure

```bash
OOD-Detection/
├── data/                   # Input data and preprocessed BPS vectors
├── models/                 # Model definitions (SCNN, LSTM, etc.)
├── training/               # Training scripts and loss functions
├── evaluation/             # Scripts for thresholding and metrics
├── utils/                  # Helper scripts (e.g., feature extraction, dataset splits)
├── results/                # Metrics, confusion matrices, and daily performance logs
└── README.md               # Project documentation
