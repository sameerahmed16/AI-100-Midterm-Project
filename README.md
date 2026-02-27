# Deep Neural Network for Diabetes Classification

**AI 100 – Midterm Project**  
**Author:** Sameer Ahmed  

## Overview

This project implements a Multi-Layer Perceptron (MLP) using PyTorch to predict diabetes from medical data. The model extends the logistic regression approach from Homework #1 by introducing hidden layers and non-linear activation functions.

The task is a binary classification problem: predicting whether a patient has diabetes (1) or not (0).

---

## Dataset

The project uses the **Pima Indians Diabetes Dataset** from the UCI Machine Learning Repository.

- 768 samples
- 8 medical features
- Target: `Outcome` (0 = No Diabetes, 1 = Diabetes)

The dataset is moderately imbalanced (approximately 65% no diabetes, 35% diabetes).

---

## Model

The implemented model is a Multi-Layer Perceptron (MLP) with:

- Input layer: 8 features
- Hidden layer 1: 16 neurons (ReLU)
- Hidden layer 2: 8 neurons (ReLU)
- Output layer: 1 neuron (Sigmoid)
- Dropout (0.2) for regularization

Training details:
- Loss function: Binary Cross Entropy (BCE)
- Optimizer: Adam (learning rate = 0.001)
- Epochs: 100
- Train/Test split: 80% / 20%
- Feature scaling using StandardScaler

---

## Results

After training:

- Test Accuracy: ~70–71%
- Evaluation metrics: Accuracy, Precision, Recall, F1-score
- Confusion matrix and training curves are generated

All outputs are saved in the `results/` directory.

---

## Project Structure
AI100-Midterm-Project/
│
├── diabetes.csv
├── train.py
├── requirements.txt
├── README.md
├── AI100_Midterm.pdf
└── results/


---

## Installation

Install dependencies:

pip install -r requirements.txt
Run the Project
python train.py

The script will:

Load and preprocess the dataset
Train the neural network
Evaluate performance
Save visualizations and the trained model

--- 

## Tools Used:
PyTorch
Scikit-learn
Pandas / NumPy
Matplotlib / Seaborn