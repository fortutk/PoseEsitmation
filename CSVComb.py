import os
import pandas as pd
from sklearn.model_selection import train_test_split

class CSV_Combiner():
    def __init__(self, input_folder: str, output_file: str):
        self.combine_csv_files(input_folder, output_file) 
        self.split_train_test(output_file, "Squat_Train.csv", "Squat_Test.csv", test_size=0.2, random_state=42) # Generate train and test CSVs

    def combine_csv_files(self, input_folder: str, output_file: str):
        """
        Combines all CSV files in a given folder into a single CSV file, adding 'Sample_ID' and 'Label' columns 
        to differentiate between individual CSV sources.
        
        Parameters:
        input_folder (str): Path to the folder containing CSV files.
        output_file (str): Path to the output combined CSV file.
        """
        all_files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
        
        if not all_files:
            print("No CSV files found in the directory.")
            return
        
        df_list = []
        for idx, file in enumerate(all_files):
            df = pd.read_csv(os.path.join(input_folder, file))
            df["Sample_ID"] = idx  # Assign a unique ID per file
            df["Label"] = os.path.splitext(file)[0]  # Extract label from filename
            df_list.append(df)
        
        combined_df = pd.concat(df_list, ignore_index=True)
        combined_df.to_csv(output_file, index=False)
        print(f"Combined CSV saved as: {output_file}")

    def split_train_test(self, input_csv: str, train_csv: str, test_csv: str, test_size: float = 0.2, random_state: int = 42):
        """
        Splits a combined CSV file into train and test CSVs while keeping samples together.
        
        Parameters:
        input_csv (str): Path to the combined CSV file.
        train_csv (str): Path to save the training CSV.
        test_csv (str): Path to save the testing CSV.
        test_size (float): Proportion of the dataset to include in the test split (default: 0.2).
        random_state (int): Random seed for reproducibility (default: 42).
        """
        df = pd.read_csv(input_csv)
        
        if "Sample_ID" not in df.columns:
            raise ValueError("The input CSV must contain a 'Sample_ID' column to group data.")
        
        unique_samples = df["Sample_ID"].unique()
        train_samples, test_samples = train_test_split(unique_samples, test_size=test_size, random_state=random_state)
        
        train_df = df[df["Sample_ID"].isin(train_samples)]
        test_df = df[df["Sample_ID"].isin(test_samples)]
        
        train_df.to_csv(train_csv, index=False)
        test_df.to_csv(test_csv, index=False)
        
        print(f"Train CSV saved as: {train_csv}")
        print(f"Test CSV saved as: {test_csv}")

# if __name__ == "__main__":
#     # Example usage
#     input_folder = "PoseCSVs"  # Replace with your folder path
#     output_file = "Squat_Full.csv"  # Replace with desired output file name
#     combine_csv_files(input_folder, output_file) 
#     split_train_test(output_file, "Squat_Train.csv", "Squat_Test.csv", test_size=0.2, random_state=42) # Generate train and test CSVs
