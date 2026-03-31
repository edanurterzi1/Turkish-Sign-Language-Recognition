<div align="center">

# 🤟 Turkish Sign Language Recognition

**A Real-Time Sign Language Recognition System Built with PyTorch**

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=for-the-badge&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-%23FF6F00.svg?style=for-the-badge&logo=TensorFlow&logoColor=white)](https://tensorflow.org/)
[![MediaPipe](https://img.shields.io/badge/MediaPipe-black?style=for-the-badge&logo=google&logoColor=white)](https://google.github.io/mediapipe/)
[![OpenCV](https://img.shields.io/badge/OpenCV-%23white.svg?style=for-the-badge&logo=opencv&logoColor=white)](https://opencv.org/)

</div>

### 📖 Overview
This project is a Deep Learning model that recognizes Turkish Sign Language using the "TR Sign Language" dataset from Kaggle. The codebase includes architectures designed using both **PyTorch** (focused on modular, object-oriented, and real-time detection) and **TensorFlow** (focused on rapid prototyping and data analysis).

### ✨ Features
- **Real-Time Detection:** Live webcam inference using MediaPipe for accurate hand tracking.
- **Modular Architecture:** Clean and extensible codebase in PyTorch for data loading, training, and testing pipelines.
- **Automated Reporting:** Automatic generation of performance metrics and sample prediction plots.
- **Alternative TensorFlow Experience:** A Jupyter Notebook built with Keras is also included for prototyping and offering a different framework perspective.

### 📊 Dataset
The models are trained strictly on the **TR Sign Language Dataset** available on Kaggle.  
👉 [Download the Dataset Here](https://www.kaggle.com/datasets/berkaykocaoglu/tr-sign-language)

> **Note:** It is recommended to place the downloaded dataset into the `data/` folder in the root directory.

### 📁 Project Structure
```text
.
├── pytorch/                 # PyTorch version (Modular & Real-time)
│   ├── main.py              # Main flow: Training, testing, evaluation graphs
│   ├── live_demo.py         # Real-time webcam detection script
│   ├── real.py              # Alternative version of live detection
│   ├── models/              # Trained model weights (.pth)
│   ├── plots/               # Evaluation metrics and sample outputs
│   └── sign_language/       # Core components (DataLoader, Model, Trainer, etc.)
│
├── tensorflow/              # TensorFlow version
│   ├── sign_language_tensorflow.ipynb  # Data analysis and model training notebook
│   └── sign_languange_model.keras      # Exported TensorFlow model
│
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation (This file)
```

### 🚀 Getting Started

**1. Install Requirements**  
It is recommended to use a virtual environment. Install dependencies with:
```bash
pip install -r requirements.txt
```

**2. Train & Evaluate the Core PyTorch Model**  
To compile the model from scratch in your own environment and generate performance plots:
```bash
cd pytorch
python main.py
```

**3. Real-Time Live Demo (Webcam)**  
To test the trained model live using your computer's webcam:
```bash
cd pytorch
python live_demo.py
```
*(Press `ESC` to exit the camera feed.)*
