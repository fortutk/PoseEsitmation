import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

class SquatPhaseDataset(Dataset):
    def __init__(self, csv_path, seq_length=30, min_velocity=0.025, sigma=2.0):
        """
        Enhanced squat phase detection dataset with:
        - Robust angle calculations
        - Physical constraints
        - Adaptive sequence labeling
        - Built-in diagnostics
        
        Args:
            csv_path: Path to CSV file
            seq_length: Sequence length in frames (default 30)
            min_velocity: Velocity threshold for phase detection (default 0.025)
            sigma: Smoothing factor (default 2.0)
        """
        self.seq_length = seq_length
        self.min_velocity = min_velocity
        self.sigma = sigma
        self.scaler = StandardScaler()
        
        # Load and preprocess data
        self.data, self.labels = self._load_and_preprocess(csv_path)
        
        # Store original stats for reference
        self.raw_means = np.mean(self.data, axis=(0,1))
        self.raw_stds = np.std(self.data, axis=(0,1))
        
        # Apply normalization
        self._normalize_data()

    def _calculate_angles(self, df):
        """Calculate biomechanically valid joint angles"""
        angles = []
        for (sample_id, time), frame in df.groupby(['Sample_ID', 'Time']):
            try:
                # Get required joints with validation
                joints = {}
                for jid in [11, 12, 23, 24, 25, 26, 27, 28]:  # Shoulders, hips, knees, ankles
                    joint_data = frame[frame['ID'] == jid]
                    if len(joint_data) == 0:
                        raise ValueError(f"Missing joint {jid}")
                    joints[jid] = joint_data[['World_X','World_Y','World_Z']].values[0]

                # Calculate vectors
                left_thigh = joints[23] - joints[25]
                left_shin = joints[27] - joints[25]
                torso = (joints[11] + joints[12])/2 - (joints[23] + joints[24])/2

                # Physically constrained angle calculation
                def safe_angle(v1, v2, min_deg=10, max_deg=170):
                    cos = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-8)
                    return np.clip(np.arccos(cos), np.radians(min_deg), np.radians(max_deg))

                angles.append({
                    'time': time,
                    'sample_id': sample_id,
                    'left_knee': safe_angle(left_thigh, left_shin, 60, 180),
                    'right_knee': safe_angle(joints[24]-joints[26], joints[28]-joints[26], 60, 180),
                    'left_hip': safe_angle(torso, left_thigh, 10, 90),
                    'torso_lean': np.arctan2(torso[1], abs(torso[2]))  # Forward lean only
                })
            except Exception as e:
                continue
        return pd.DataFrame(angles)

    def _create_sequences(self, angle_df):
        """Create sequences with adaptive labeling for squat phases"""
        sequences = []
        labels = []
        angle_cols = ['left_knee', 'right_knee', 'left_hip', 'torso_lean']
        
        for sample_id, group in angle_df.groupby('sample_id'):
            group = group.sort_values('time')
            angles = group[angle_cols].values
            
            # Apply feature-specific smoothing
            smoothed = np.array([gaussian_filter1d(angles[:,i], 
                               sigma=self.sigma*(1+i*0.2))  # More smoothing for higher indices
                              for i in range(len(angle_cols))]).T
            
            # Calculate velocity with enhanced method
            velocity = np.gradient(smoothed[:,0])
            
            # Create sequences with 66% overlap
            for i in range(0, len(angles)-self.seq_length+1, self.seq_length//3):
                window = smoothed[i:i+self.seq_length]
                
                # Enhanced labeling criteria
                knee_angle_change = window[-1, 0] - window[0, 0]  # Change in knee angle
                hip_angle_change = window[-1, 2] - window[0, 2]    # Change in hip angle
                torso_lean_avg = np.mean(window[:, 3])               # Average torso lean
                mean_vel = np.mean(velocity[i:i+self.seq_length])
                
                # Phase detection logic
                # Label 1 for "UP" if knee angle change is significant and velocity is high
                if knee_angle_change > 0.05 and mean_vel > self.min_velocity and torso_lean_avg < 0.5:
                    label = 1  # "UP"
                # Label 0 for "DOWN" if velocity is negative and no significant upward movement
                elif mean_vel < -self.min_velocity and knee_angle_change < -0.05:
                    label = 0  # "DOWN"
                # Label 2 for "STABLE" if there is no significant movement
                else:
                    label = 2  # "STABLE"
                
                sequences.append(window)
                labels.append(label)
                
        return np.array(sequences), np.array(labels)

    def _load_and_preprocess(self, csv_path):
        """Full preprocessing pipeline"""
        df = pd.read_csv(csv_path)
        angle_df = self._calculate_angles(df)
        return self._create_sequences(angle_df)

    def _normalize_data(self):
        """Normalize while preserving joint relationships"""
        orig_shape = self.data.shape
        flattened = self.data.reshape(-1, orig_shape[-1])
        self.data = self.scaler.fit_transform(flattened).reshape(orig_shape)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.FloatTensor(self.data[idx]),
            torch.LongTensor([self.labels[idx]])
        )

    def collate_fn(self, batch):
        sequences, labels = zip(*batch)
        return pad_sequence(sequences, batch_first=True), torch.cat(labels)

    def analyze(self):
        """Comprehensive dataset analysis"""
        print(f"\n{' Dataset Analysis ':=^80}")
        print(f"Total sequences: {len(self.data)}")
        print(f"Class balance: DOWN {sum(self.labels==0)} | UP {sum(self.labels==1)} | STABLE {sum(self.labels==2)}")
        print(f"Raw means (deg): {np.rad2deg(self.raw_means)}")
        print(f"Raw stds (deg): {np.rad2deg(self.raw_stds)}")
        
        # Plot distributions
        plt.figure(figsize=(12,4))
        plt.subplot(121)
        plt.hist(self.labels, bins=3, rwidth=0.8)
        plt.xticks([0.25, 1.25, 2.25], ['DOWN', 'UP', 'STABLE'])
        plt.title('Class Distribution')
        
        plt.subplot(122)
        for i, name in enumerate(['Left Knee','Right Knee','Left Hip','Torso Lean']):
            plt.hist(np.rad2deg(self.data[:,:,i].ravel()), 
                    bins=50, alpha=0.5, label=name)
        plt.title('Feature Distributions')
        plt.xlabel('Angle (degrees)')  # Added X-axis label
        plt.ylabel('Frequency')  # Added Y-axis label
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    # Initialize dataset
    dataset = SquatPhaseDataset(
        "Squat_Test.csv",
        seq_length=30,
        min_velocity=0.025,
        sigma=2.0
    )
    
    # Analyze dataset
    dataset.analyze()
    
    # Create balanced dataloader
    def create_balanced_loader(dataset, batch_size=32):
        class_weights = 1. / np.bincount(dataset.labels)
        sample_weights = class_weights[dataset.labels]
        sampler = torch.utils.data.WeightedRandomSampler(
            sample_weights, len(sample_weights), replacement=True)
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            collate_fn=dataset.collate_fn
        )
    
    train_loader = create_balanced_loader(dataset)
    batch = next(iter(train_loader))
    print(f"\nBatch shapes - Sequences: {batch[0].shape}, Labels: {batch[1].shape}")
