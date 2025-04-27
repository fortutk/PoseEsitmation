import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from FrameDataset import SquatKneeFrameDataset

class SquatMLP(nn.Module):
    def __init__(self, input_dim, hidden_dims=[64, 32], dropout=0.3):
        super(SquatMLP, self).__init__()
        layers = []
        last_dim = input_dim
        for h in hidden_dims:
            layers.append(nn.Linear(last_dim, h))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            last_dim = h
        layers.append(nn.Linear(last_dim, 2))  # Output: 2 classes (DOWN, UP)
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

def train(model, dataloader, epochs=10, lr=1e-3):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for features, labels in dataloader:
            features, labels = features.to(device), labels.squeeze(1).to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item() * features.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        acc = correct / total
        print(f"Epoch {epoch+1}/{epochs} | Loss: {total_loss/total:.4f} | Accuracy: {acc:.2%}")

def evaluate(model, dataloader, return_preds=False):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for features, labels in dataloader:
            features = features.to(device)
            labels = labels.squeeze(1).to(device)

            outputs = model(features)
            preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Print classification report
    print("\nClassification Report:")
    print(classification_report(all_labels, all_preds, target_names=['DOWN', 'UP']))

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['DOWN', 'UP'], yticklabels=['DOWN', 'UP'])
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.show()

    if return_preds:
        return np.array(all_preds), np.array(all_labels)


if __name__ == "__main__":
    dataset = SquatKneeFrameDataset("Squat_Train.csv", threshold_pct=50, sigma=2.0)
    test_dataset = SquatKneeFrameDataset("Squat_Test.csv", threshold_pct=50, sigma=2.0)
    dataset.analyze()

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    input_dim = dataset[0][0].shape[0]
    model = SquatMLP(input_dim=input_dim)

    train(model, loader, epochs=15, lr=0.001)
    evaluate(model, test_loader, return_preds=True)
