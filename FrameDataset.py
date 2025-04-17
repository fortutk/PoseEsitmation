import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from scipy.ndimage import gaussian_filter1d
import matplotlib.pyplot as plt

class SquatKneeFrameDataset(Dataset):
    def __init__(self, csv_path, threshold_pct=50, sigma=1.5):
        """
        Frame-wise dataset labeling UP or DOWN based on both knee angles.
        
        Args:
            csv_path: Path to CSV
            threshold_pct: Percentile used as the UP/DOWN threshold (default median)
            sigma: Smoothing factor
        """
        self.threshold_pct = threshold_pct
        self.sigma = sigma
        self.scaler = StandardScaler()

        self.data, self.labels = self._load_and_preprocess(csv_path)
        self._normalize_data()

        self.raw_means = np.mean(self.data, axis=0)
        self.raw_stds = np.std(self.data, axis=0)

    def _calculate_angles(self, df):
        angles = []
        for (sample_id, time), frame in df.groupby(['Sample_ID', 'Time']):
            try:
                joints = {}
                for jid in [11, 12, 23, 24, 25, 26, 27, 28]:
                    joint_data = frame[frame['ID'] == jid]
                    if len(joint_data) == 0:
                        raise ValueError(f"Missing joint {jid}")
                    joints[jid] = joint_data[['World_X','World_Y','World_Z']].values[0]

                left_thigh = joints[23] - joints[25]
                left_shin = joints[27] - joints[25]
                right_thigh = joints[24] - joints[26]
                right_shin = joints[28] - joints[26]
                torso = (joints[11] + joints[12])/2 - (joints[23] + joints[24])/2

                def safe_angle(v1, v2):
                    cos = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-8)
                    return np.clip(np.arccos(cos), 0, np.pi)

                angles.append({
                    'time': time,
                    'sample_id': sample_id,
                    'left_knee': safe_angle(left_thigh, left_shin),
                    'right_knee': safe_angle(right_thigh, right_shin),
                    'left_hip': safe_angle(torso, left_thigh),
                    'torso_lean': np.arctan2(torso[1], abs(torso[2]))
                })
            except:
                continue
        return pd.DataFrame(angles)

    def _label_all_frames(self, angle_df):
        angle_df = angle_df.sort_values(['sample_id', 'time'])

        # Features
        features = angle_df[['left_knee', 'right_knee', 'left_hip', 'torso_lean']].values
        smoothed = gaussian_filter1d(features, sigma=self.sigma, axis=0)

        # Knee-based UP/DOWN classification
        mean_knee_angle = (smoothed[:, 0] + smoothed[:, 1]) / 2
        threshold = np.percentile(mean_knee_angle, self.threshold_pct)

        labels = (mean_knee_angle >= threshold).astype(int)

        print(f"Threshold (rad): {threshold:.2f} | (deg): {np.rad2deg(threshold):.1f}")
        print(f"Labeled as UP: {np.sum(labels == 1)}, DOWN: {np.sum(labels == 0)}")

        return smoothed, labels

    def _load_and_preprocess(self, csv_path):
        df = pd.read_csv(csv_path)
        angle_df = self._calculate_angles(df)
        return self._label_all_frames(angle_df)

    def _normalize_data(self):
        self.data = self.scaler.fit_transform(self.data)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return torch.FloatTensor(self.data[idx]), torch.LongTensor([self.labels[idx]])

    def analyze(self):
        print(f"\n{' Frame-wise Dataset Analysis ':=^80}")
        print(f"Total labeled frames: {len(self.data)}")
        print(f"Class balance: DOWN {sum(self.labels==0)} | UP {sum(self.labels==1)}")
        print(f"Raw means (deg): {np.rad2deg(self.raw_means)}")
        print(f"Raw stds (deg): {np.rad2deg(self.raw_stds)}")

        plt.figure(figsize=(12,4))
        plt.subplot(121)
        plt.hist(self.labels, bins=[-0.5, 0.5, 1.5], rwidth=0.8)
        plt.xticks([0, 1], ['DOWN', 'UP'])
        plt.title('Class Distribution')

        plt.subplot(122)
        for i, name in enumerate(['Left Knee','Right Knee','Left Hip','Torso Lean']):
            plt.hist(np.rad2deg(self.data[:,i]), bins=50, alpha=0.5, label=name)
        plt.title('Angle Feature Distributions')
        plt.xlabel('Angle (degrees)')
        plt.ylabel('Frequency')
        plt.legend()
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    dataset = SquatKneeFrameDataset(
        "Squat_Test.csv",
        threshold_pct=50,  # median
        sigma=2.0
    )

    dataset.analyze()

    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    batch = next(iter(loader))
    print(f"\nBatch - Features: {batch[0].shape}, Labels: {batch[1].shape}")
