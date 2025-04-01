import torch
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

class SquatPosePCADataset(Dataset):
    def __init__(self, csv_path, seq_length=30, variance_threshold=0.95):
        """
        Args:
            csv_path (str): Path to CSV file
            seq_length (int): Number of frames per sequence
            variance_threshold (float): Keep components until this variance is explained (0-1)
        """
        self.seq_length = seq_length
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.pca = PCA(n_components=variance_threshold)  # Auto-select components
        
        self.data, self.labels = self._load_and_preprocess(csv_path)
        self.labels = self.label_encoder.fit_transform(self.labels)

    def _load_and_preprocess(self, csv_path):
        df = pd.read_csv(csv_path)
        
        # 1. Use ALL coordinate systems
        coord_systems = {
            'pixel': ['Pixel_X', 'Pixel_Y'],
            'norm': ['Norm_X', 'Norm_Y', 'Norm_Z'], 
            'world': ['World_X', 'World_Y', 'World_Z']
        }
        
        # 2. Create all possible features (33 joints × 8 coordinates)
        features = []
        for sys_name, coords in coord_systems.items():
            for jid in range(33):
                for coord in coords:
                    col_name = f"{sys_name}_{jid}_{coord.split('_')[-1].lower()}"
                    features.append(col_name)
        
        # 3. Pivot to get all features per frame
        pivot_df = df.pivot(index=['Sample_ID', 'Time'], columns='ID', 
                          values=[c for sys in coord_systems.values() for c in sys])
        
        # 4. Flatten columns
        pivot_df.columns = features
        
        # 5. Normalize and apply PCA
        X = self.scaler.fit_transform(pivot_df[features])
        X_pca = self.pca.fit_transform(X)
        
        print(f"Original features: {len(features)}")
        print(f"Reduced to {self.pca.n_components_} components")
        print(f"Explained variance: {sum(self.pca.explained_variance_ratio_):.3f}")
        
        # 6. Create reduced DataFrame
        reduced_df = pd.DataFrame(
            X_pca,
            columns=[f"pca_{i}" for i in range(X_pca.shape[1])],
            index=pivot_df.index
        )
        
        # 7. Add labels and create sequences
        labels = df.groupby('Sample_ID')['Label'].first()
        reduced_df = reduced_df.join(labels, on='Sample_ID')
        
        return self._create_sequences(reduced_df)

    def _create_sequences(self, df):
        sequences, labels = [], []
        feature_cols = [c for c in df.columns if c.startswith('pca_')]
        
        for sample_id, group in df.groupby('Sample_ID'):
            video_data = group[feature_cols].values
            
            # Create sequences with 25% overlap
            stride = self.seq_length // 4 or 1
            for i in range(0, len(video_data) - self.seq_length + 1, stride):
                seq = video_data[i:i + self.seq_length]
                sequences.append(seq)
                labels.append(group['Label'].iloc[0])
        
        return sequences, labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return (
            torch.tensor(self.data[idx], dtype=torch.float32),  # [seq_len, n_components]
            torch.tensor(self.labels[idx], dtype=torch.long)    # [1]
        )

    def collate_fn(self, batch):
        sequences, labels = zip(*batch)
        padded_seqs = pad_sequence(sequences, batch_first=True, padding_value=0)
        return padded_seqs, torch.stack(labels)
    
def visualize_pca_results(dataset, feature_names, n_top_features=3):
    """
    Fixed visualization for PCA results
    """
    pca = dataset.pca
    
    # 1. Variance Explained Plot
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.bar(range(pca.n_components_), pca.explained_variance_ratio_)
    plt.xlabel('Principal Component')
    plt.ylabel('Variance Explained')
    plt.title(f'Total Variance Explained: {sum(pca.explained_variance_ratio_):.2%}')
    
    # 2. Cumulative Variance
    plt.subplot(1, 2, 2)
    plt.plot(np.cumsum(pca.explained_variance_ratio_))
    plt.xlabel('Number of Components')
    plt.ylabel('Cumulative Explained Variance')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # 3. Top Features per Component (as text)
    print("\nTop Contributing Features per Principal Component:")
    components = np.abs(pca.components_)
    for i in range(pca.n_components_):
        top_indices = np.argsort(-components[i])[:n_top_features]
        print(f"PC_{i}:")
        for idx in top_indices:
            print(f"  - {feature_names[idx]} (loading: {components[i, idx]:.3f})")
    
    # 4. Component Heatmap (first 10 features for visibility)
    plt.figure(figsize=(12, 6))
    sns.heatmap(
        components[:, :10],  # First 10 features only
        cmap='viridis',
        annot=True,
        fmt=".2f",
        yticklabels=[f"PC_{i}" for i in range(pca.n_components_)],
        xticklabels=feature_names[:10]
    )
    plt.title('PCA Component Loadings (First 10 Features)')
    plt.show()

# Usage
if __name__ == "__main__":
    dataset = SquatPosePCADataset(
        "Squat_Test.csv",
        seq_length=30,
        variance_threshold=0.95  # Keep components until 95% variance explained
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=32,
        collate_fn=dataset.collate_fn
    )
    
    # Verify
    for batch in dataloader:
        seqs, labels = batch
        print(f"Batch shape: {seqs.shape}")  # [32, 30, n_components]
        print("PCA components:", dataset.pca.n_components_)
        break
    
    dataset = SquatPosePCADataset("Squat_Test.csv", seq_length=30)
    
    # Generate feature names (replace with your actual column names)
    coord_systems = {
        'pixel': ['Pixel_X', 'Pixel_Y'],
        'norm': ['Norm_X', 'Norm_Y', 'Norm_Z'], 
        'world': ['World_X', 'World_Y', 'World_Z']
    }
    
    feature_names = [
        f"{sys_name}_{jid}_{coord.split('_')[-1].lower()}"
        for sys_name, coords in coord_systems.items()
        for jid in range(33)
        for coord in coords
    ]
    
    # Visualize
    visualize_pca_results(dataset, feature_names)