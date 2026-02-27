Deep Neural Network for Diabetes Classification

AI 100 – Midterm Project
Author: Sameer Ahmed


Overview

This project implements a Multi-Layer Perceptron (MLP) using PyTorch to predict diabetes from medical data. The model extends the logistic regression approach from Homework #1 by introducing hidden layers and non-linear activation functions.

The task is a binary classification problem: predicting whether a patient has diabetes (1) or not (0).


Dataset

The project uses the Pima Indians Diabetes Dataset from the UCI Machine Learning Repository.

- 768 samples
- 8 medical features
- Target variable: Outcome (0 = No Diabetes, 1 = Diabetes)

The dataset is moderately imbalanced (about 65% no diabetes and 35% diabetes).


Model

The implemented model is a Multi-Layer Perceptron (MLP) with:

- Input layer: 8 features
- Hidden layer 1: 16 neurons (ReLU activation)
- Hidden layer 2: 8 neurons (ReLU activation)
- Output layer: 1 neuron (Sigmoid activation)
- Dropout (0.2) for regularization

Training setup:

- Loss function: Binary Cross Entropy (BCE)
- Optimizer: Adam (learning rate = 0.001)
- Epochs: 100
- Train/Test split: 80% / 20%
- Feature scaling using StandardScaler


Results

After training, the model achieves approximately:

- Test Accuracy: 70–71%
- Evaluation metrics: Accuracy, Precision, Recall, F1-score

Confusion matrix, training curves, performance metrics, and the trained model are saved in the "results" directory.


Project Structure

AI100-Midterm-DeepLearning/

- diabetes.csv
- train.py
- requirements.txt
- README.md
- results/

Installation

Install dependencies:

pip install -r requirements.txt


Run the Project

python train.py

The script will:

1. Load and preprocess the dataset
2. Train the neural network
3. Evaluate performance on the test set
4. Save visualizations and the trained model


Tools Used

- PyTorch
- Scikit-learn
- Pandas and NumPy
- Matplotlib and Seaborn
