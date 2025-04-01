import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d

class SquatPhaseDataset(Dataset):
    def __init__(self, csv_path, seq_length=30, min_velocity=0.05):
        """
        Args:
            csv_path: Path to CSV file
            seq_length: Number of frames per sequence
            min_velocity: Minimum angular velocity threshold for phase detection
        """
        self.seq_length = seq_length
        self.min_velocity = min_velocity
        self.scaler = StandardScaler()
        self.data, self.labels = self._load_and_preprocess(csv_path)

    def _calculate_angles(self, df):
        """Calculate joint angles with smoothing"""
        angles = []
        for (sample_id, time), frame_data in df.groupby(['Sample_ID', 'Time']):
            # Get joint coordinates (adjust indices as needed)
            hip = frame_data[frame_data['ID'] == 23][['World_X', 'World_Y', 'World_Z']].values[0]
            knee = frame_data[frame_data['ID'] == 25][['World_X', 'World_Y', 'World_Z']].values[0]
            ankle = frame_data[frame_data['ID'] == 27][['World_X', 'World_Y', 'World_Z']].values[0]
            
            # Calculate vectors
            thigh = hip - knee
            shin = ankle - knee
            
            # Calculate knee angle
            cos_theta = np.dot(thigh, shin) / (np.linalg.norm(thigh) * np.linalg.norm(shin))
            knee_angle = np.arccos(np.clip(cos_theta, -1, 1))
            
            angles.append({
                'Sample_ID': sample_id,
                'Time': time,
                'knee_angle': knee_angle
            })
        return pd.DataFrame(angles)

    def _create_sequences(self, angle_df):
        """Create sequences with phase labels"""
        sequences, labels = [], []
        
        for sample_id, group in angle_df.groupby('Sample_ID'):
            angles = group['knee_angle'].values
            if len(angles) < self.seq_length:
                continue
                
            # Apply Gaussian smoothing
            smoothed = gaussian_filter1d(angles, sigma=2)
            velocity = np.gradient(smoothed)
            
            # Create sequences with 50% overlap
            for i in range(0, len(angles)-self.seq_length+1, self.seq_length//2):
                seq = angles[i:i+self.seq_length]
                vel_seq = velocity[i:i+self.seq_length]
                
                # Label based on majority velocity
                label = 1 if np.mean(vel_seq) > self.min_velocity else 0
                sequences.append(seq)
                labels.append(label)
                
        return np.array(sequences), np.array(labels)

    def _load_and_preprocess(self, csv_path):
        df = pd.read_csv(csv_path)
        angle_df = self._calculate_angles(df)
        angle_df['knee_angle'] = self.scaler.fit_transform(angle_df[['knee_angle']])
        return self._create_sequences(angle_df)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32).unsqueeze(-1),  # [seq_len, 1]
            torch.tensor(self.labels[idx], dtype=torch.long)                   # [1]
        )

    def collate_fn(self, batch):
        sequences, labels = zip(*batch)
        padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)
        return padded_seqs, torch.stack(labels)

def plot_sequence_with_labels(dataset, n_samples=3):
    """Plot sequences with phase labels"""
    plt.figure(figsize=(15, 5))
    for i in range(n_samples):
        seq = dataset.data[i]
        label = dataset.labels[i]
        
        plt.subplot(1, n_samples, i+1)
        plt.plot(seq, label='Knee Angle')
        plt.title(f"Label: {'Up' if label else 'Down'}")
        plt.xlabel('Frame')
        plt.ylabel('Normalized Angle')
        plt.axhline(0, color='k', linestyle='--')
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    # Initialize dataset
    dataset = SquatPhaseDataset(
        "Squat_Test.csv",
        seq_length=30,
        min_velocity=0.05  # Adjust based on your data
    )
    
    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=dataset.collate_fn,
        shuffle=True
    )
    
    # Verify batch
    for seqs, labels in dataloader:
        print(f"Batch shape: {seqs.shape}")  # [batch_size, seq_len, 1]
        print(f"Sample labels: {labels[:5]}")
        break
    
    # Visualize
    plot_sequence_with_labels(dataset)