import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt
from Dataset import SquatPosePCADataset

# ====================== Configuration ======================
CONFIG = {
    'hidden_size': 64,
    'num_layers': 1,
    'lr': 0.001,
    'batch_size': 32,
    'epochs': 50,
    'num_pca_components': 3,  # Use first 3 PCA components instead of just 1
    'phase_threshold': 0.1
}

# ====================== Dynamic LSTM Model ======================
class PhaseLSTM(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        self.input_size = input_size
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=CONFIG['hidden_size'],
            num_layers=CONFIG['num_layers'],
            batch_first=True
        )
        self.classifier = nn.Sequential(
            nn.Linear(CONFIG['hidden_size'], 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = x.float()
        out, _ = self.lstm(x)
        return self.classifier(out[:, -1]).squeeze()

# ====================== Data Preparation ======================
def create_phase_labels(sequences):
    labels = []
    for seq in sequences:
        if len(seq) < 2:
            continue
        # Use multiple PCA components to determine phase
        dominant_motion = seq[:, :CONFIG['num_pca_components']]  # Take first N components
        # Compute gradient magnitude across components
        grads = np.array([np.gradient(comp) for comp in dominant_motion.T])
        grad_magnitude = np.linalg.norm(grads, axis=0)  # Euclidean norm of gradients
        # Determine direction using first component (assuming it's still the most important)
        direction = np.sign(grads[0])  # Sign of the first component's gradient
        phase = int(np.mean(grad_magnitude * direction > CONFIG['phase_threshold']) > 0.5)
        labels.append(phase)
    return torch.tensor(labels, dtype=torch.float32)

def prepare_loaders():
    train_set = SquatPosePCADataset("Squat_Train.csv")
    val_set = SquatPosePCADataset("Squat_Test.csv")

    # Get actual feature dimensions
    train_feat_dim = train_set.data[0].shape[1]
    val_feat_dim = val_set.data[0].shape[1]
    print(f"Training features: {train_feat_dim}, Validation features: {val_feat_dim}")

    # Ensure we have enough components
    feat_dim = min(train_feat_dim, val_feat_dim, CONFIG['num_pca_components'])
    CONFIG['input_size'] = feat_dim  # Update config

    # Trim features to use first N PCA components
    train_seqs = [torch.tensor(seq[:, :feat_dim], dtype=torch.float32) 
                 for seq in train_set.data if len(seq) >= 2]
    val_seqs = [torch.tensor(seq[:, :feat_dim], dtype=torch.float32)
               for seq in val_set.data if len(seq) >= 2]

    train_labels = create_phase_labels([seq.numpy() for seq in train_seqs])
    val_labels = create_phase_labels([seq.numpy() for seq in val_seqs])

    train_loader = DataLoader(
        list(zip(train_seqs, train_labels)),
        batch_size=CONFIG['batch_size'],
        shuffle=True
    )
    val_loader = DataLoader(
        list(zip(val_seqs, val_labels)),
        batch_size=CONFIG['batch_size']
    )
    return train_loader, val_loader

# ====================== Training ======================
def train_model(train_loader, val_loader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = PhaseLSTM(CONFIG['input_size']).to(device)
    optimizer = optim.Adam(model.parameters(), lr=CONFIG['lr'])
    criterion = nn.BCELoss()

    best_acc = 0
    for epoch in range(CONFIG['epochs']):
        model.train()
        train_loss = 0
        for seq, labels in train_loader:
            seq, labels = seq.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(seq)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()

        val_acc = evaluate(model, val_loader, device)
        print(f"Epoch {epoch+1}/{CONFIG['epochs']}: "
              f"Loss: {train_loss/len(train_loader):.4f}, "
              f"Val Acc: {val_acc:.4f}")

        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), "best_model.pth")

    print(f"\nBest validation accuracy: {best_acc:.4f}")
    return model

def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for seq, labels in loader:
            seq = seq.to(device)
            preds = (model(seq).cpu() > 0.5).float()
            correct += (preds == labels.cpu()).sum().item()
            total += len(labels)
    return correct / total

# ====================== Main ======================
if __name__ == "__main__":
    train_loader, val_loader = prepare_loaders()
    print(f"Using {CONFIG['input_size']} PCA components")
    
    sample_seq, _ = next(iter(train_loader))
    print(f"Sample seq shape: {sample_seq.shape}")
    
    model = train_model(train_loader, val_loader)
    final_acc = evaluate(model, val_loader, 
                        device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    print(f"\nFinal model accuracy: {final_acc:.4f}")