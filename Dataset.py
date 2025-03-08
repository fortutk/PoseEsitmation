
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

class Pose_Dataset(Dataset):

    def __init__(self, scale=True, train=True):
        self.X, self.y = self.load_data(train)

        # Encode string labels to numeric values
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y.values.ravel())  # Flatten to 1D array

        transform_train = transforms.Compose([
            transforms.ToTensor(),
        ])
        transform_test = transforms.Compose([
          transforms.ToTensor(),
        ])

        self.transform = transform_train if train else transform_test
        
        # Convert y into a tensor
        self.y = torch.tensor(self.y, dtype=torch.long)  

    def load_data(self, train):
        urls = ['Squat_test.csv', 'Squat_train.csv']
        url = urls[int(train)]

        data = pd.read_csv(url)

        X_cols = ["Time", "ID", "Pixel_X", "Pixel_Y", "Norm_X", "Norm_Y", "Norm_Z", "World_X", "World_Y", "World_Z"]
        y_cols = ['Label']

        X = data[X_cols].copy().astype(np.float32)  # Ensure numerical values
        y = data[y_cols]  # Keep labels as strings

        X = X.reset_index(drop=True)

        return X, y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        X = self.X.iloc[idx].values  # Retrieve row as NumPy array
        label = self.y[idx]  # Already a tensor

        X = self.tran  # Convert features to tensor

        return X, label

    def decode_labels(self, encoded_labels):
        """Convert numeric labels back to original string labels."""
        return self.label_encoder.inverse_transform(encoded_labels)
    
    def transform(self, X):
        return self.transform(X)
    
    def inverse_transform(self, labels):
        labels_unscaled = self.scaler.inverse_transform(labels)
        return labels_unscaled

class Pose_Dataloader():
    def __init__(self, train, test):
        self.train_loader = DataLoader(train, batch_size=32, shuffle=True)
        self.test_loader = DataLoader(test, batch_size=32, shuffle=False)

if __name__ == "__main__":
    train = Pose_Dataset(train=True)
    test = Pose_Dataset(train=False)
    dataloader = Pose_Dataloader(train, test)
    print(len(train), len(test))
    print(len(dataloader.train_loader), len(dataloader.test_loader))
    print(f'{next(iter(dataloader.train_loader))}')