# OOD-Detection

Code repository for the paper:  
**"YouTube Video Identification with Out-of-Distribution (OOD) Detection in Encrypted Network Traffic"**  
_Muhammad B. Sarwar, Syed M. Ahmad, Maheera Amjad, Muhammad U. S. Khan, Waqar A. Malik, Samee U. Khan_

Submitted to IEEE Access.

---

## 🔍 Overview

This repository provides a privacy-preserving framework for classifying YouTube video streams in encrypted network traffic and detecting previously unseen (out-of-distribution, OOD) traffic. It leverages deep learning architectures—SCNN, SCNN-LSTM, AlexNet, LeNet, and BPS—and applies threshold-based classification to reject unknown classes.

---

## 🧠 Core Features

- **Encrypted Traffic Classification**: Operates on metadata (e.g., BPS vectors, inter-arrival times, packet sizes), not payloads.
- **Multiple Architectures**: CNN-based and hybrid CNN-LSTM models.
- **Out-of-Distribution Detection**: Supports ROC and percentile-based thresholding.
- **Custom Loss Functions**: Combines CrossEntropy with KL divergence for better generalization.
- **Daily Evaluation**: 14-day dataset covering 30 known and 13 unseen video classes.

---

```bash
## 📁 Directory Structure
OOD-Detection/
├── DataSet/ # Raw BPS vectors for training
├── Seen_Data/ # Processed in-distribution data
├── Unseen_Data/ # Processed OOD samples for each day
├── Results/ # Evaluation outputs (graphs, metrics)
├── Threshold.py # ROC and percentile-based threshold calibration
├── graphs.py # Visualization utilitie
├── metrics.py # Precision, Recall, F1 computation
├── data_helpers.py # Feature processing and input prep
├── data_extract.py # Converts raw pcap features into BPS vectors
├── *.py # Model-specific training/inference scripts

```

---

## ⚙️ Setup & Installation

### Requirements

- Python 3.8+
- PyTorch
- numpy
- pandas
- scikit-learn
- matplotlib
- tqdm

Install dependencies:

```bash
pip install -r requirements.txt
```

## 👥 Authors
1. Muhammad B. Sarwar
2. Syed M. Ahmad
3. Maheera Amjad
4. Muhammad U. S. Khan
5. Waqar A. Malik
6. Samee U. Khan

## 📬 Contact
For questions, reach out to:
```bash
msarwar.mscse22sines@student.nust.edu.pk
```
