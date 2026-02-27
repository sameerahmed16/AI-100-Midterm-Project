"""
Deep Neural Network for Diabetes Classification
AI 100 - Midterm Project
Author: Sameer Ahmed

This script implements a Multi-Layer Perceptron (MLP) to predict diabetes
using the Pima Indians Diabetes dataset.
"""

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import os

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Create results directory if it doesn't exist
os.makedirs('results', exist_ok=True)

# Load dataset
print("Loading dataset...")
data = pd.read_csv("diabetes.csv")
print(f"Dataset shape: {data.shape}")
print(f"Features: {data.columns.tolist()[:-1]}")
print(f"Class distribution:\n{data['Outcome'].value_counts()}")

# Prepare features and labels
X = data.drop("Outcome", axis=1).values
y = data["Outcome"].values

# Normalize features (important for neural networks)
scaler = StandardScaler()
X = scaler.fit_transform(X)

# Train-test split (80-20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTraining set size: {X_train.shape[0]}")
print(f"Test set size: {X_test.shape[0]}")

# Convert to PyTorch tensors
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32).view(-1, 1)

X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32).view(-1, 1)

# Define Multi-Layer Perceptron (MLP)
class MLP(nn.Module):
    """
    Multi-Layer Perceptron for binary classification.
    
    Architecture:
    - Input layer: 8 features
    - Hidden layer 1: 16 neurons with ReLU activation
    - Hidden layer 2: 8 neurons with ReLU activation
    - Output layer: 1 neuron with Sigmoid activation
    """
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(8, 16)  # Input to first hidden layer
        self.fc2 = nn.Linear(16, 8)  # First to second hidden layer
        self.fc3 = nn.Linear(8, 1)   # Second hidden to output layer
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()
        self.dropout = nn.Dropout(0.2)  # Add dropout for regularization

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize model
model = MLP()
print(f"\nModel architecture:\n{model}")

# Count parameters
total_params = sum(p.numel() for p in model.parameters())
trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"Total parameters: {total_params}")
print(f"Trainable parameters: {trainable_params}")

# Loss function and optimizer
criterion = nn.BCELoss()  # Binary Cross Entropy Loss
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Training loop
num_epochs = 100
train_losses = []
train_accuracies = []

print("\nStarting training...")
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Calculate training accuracy
    with torch.no_grad():
        predictions = (outputs > 0.5).float()
        accuracy = accuracy_score(y_train.numpy(), predictions.numpy())
    
    train_losses.append(loss.item())
    train_accuracies.append(accuracy)
    
    # Print progress every 10 epochs
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.4f}, Accuracy: {accuracy:.4f}")

print("\nTraining completed!")

# Evaluation on test set
print("\nEvaluating on test set...")
model.eval()
with torch.no_grad():
    test_outputs = model(X_test)
    test_predictions = (test_outputs > 0.5).float()
    test_accuracy = accuracy_score(y_test.numpy(), test_predictions.numpy())
    test_loss = criterion(test_outputs, y_test).item()

print(f"Test Accuracy: {test_accuracy:.4f} ({test_accuracy*100:.2f}%)")
print(f"Test Loss: {test_loss:.4f}")

# Classification report
print("\nClassification Report:")
print(classification_report(y_test.numpy(), test_predictions.numpy(), 
                          target_names=['No Diabetes', 'Diabetes']))

# Confusion Matrix
cm = confusion_matrix(y_test.numpy(), test_predictions.numpy())
print("\nConfusion Matrix:")
print(cm)

# Visualizations
print("\nGenerating visualizations...")

# 1. Training Loss and Accuracy
plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(train_losses, label='Training Loss', color='blue')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(1, 2, 2)
plt.plot(train_accuracies, label='Training Accuracy', color='green')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training Accuracy Over Epochs')
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('results/training_curves.png', dpi=300, bbox_inches='tight')
print("Saved: results/training_curves.png")

# 2. Confusion Matrix Heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=['No Diabetes', 'Diabetes'],
            yticklabels=['No Diabetes', 'Diabetes'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.savefig('results/confusion_matrix.png', dpi=300, bbox_inches='tight')
print("Saved: results/confusion_matrix.png")

# 3. Model Performance Summary
plt.figure(figsize=(10, 6))
metrics = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
from sklearn.metrics import precision_score, recall_score, f1_score

precision = precision_score(y_test.numpy(), test_predictions.numpy())
recall = recall_score(y_test.numpy(), test_predictions.numpy())
f1 = f1_score(y_test.numpy(), test_predictions.numpy())

values = [test_accuracy, precision, recall, f1]
colors = ['#3498db', '#2ecc71', '#e74c3c', '#f39c12']

bars = plt.bar(metrics, values, color=colors, alpha=0.7, edgecolor='black')
plt.ylabel('Score', fontsize=12)
plt.title('Model Performance Metrics', fontsize=14, fontweight='bold')
plt.ylim([0, 1])
plt.grid(True, alpha=0.3, axis='y')

# Add value labels on bars
for bar, value in zip(bars, values):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
             f'{value:.3f}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('results/performance_metrics.png', dpi=300, bbox_inches='tight')
print("Saved: results/performance_metrics.png")

# 4. Feature importance visualization (using model weights)
plt.figure(figsize=(10, 6))
feature_names = data.columns.tolist()[:-1]  # Exclude Outcome

# Get weights from first layer
first_layer_weights = model.fc1.weight.data.numpy()
feature_importance = np.abs(first_layer_weights).mean(axis=0)

plt.barh(feature_names, feature_importance, color='steelblue', alpha=0.7, edgecolor='black')
plt.xlabel('Average Absolute Weight', fontsize=12)
plt.title('Feature Importance (First Layer Weights)', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='x')
plt.tight_layout()
plt.savefig('results/feature_importance.png', dpi=300, bbox_inches='tight')
print("Saved: results/feature_importance.png")

print("\n" + "="*50)
print("All visualizations saved to results/ directory")
print("="*50)

# Save model
torch.save(model.state_dict(), 'results/model.pth')
print("\nModel saved to: results/model.pth")

