import pandas as pd
import numpy as np
from sklearn.utils import shuffle
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description='Split LJSpeech dataset into train and val sets.')
parser.add_argument('--dataset_path', type=str, required=True, help='Path to LJSpeech metadata CSV file')
args = parser.parse_args()

# Load the LJSpeech dataset
df = pd.read_csv(os.path.join(args.dataset_path, "metadata.csv"), sep='|', header=None)  # Assuming '|' separator and no header

# Shuffle the dataset
df_shuffled = shuffle(df, random_state=42)

# Split into validation (first 100) and training (rest)
val_df = df_shuffled[:100]
train_df = df_shuffled[100:]

train_paths = args.dataset_path + '/wavs/' + train_df[0]
val_paths = args.dataset_path + '/wavs/' + val_df[0]

# Write file paths (first column) to val.txt and train.txt in the dataset directory
val_paths.to_csv(os.path.join("dataset_raw", 'val'), index=False, header=False)
train_paths.to_csv(os.path.join("dataset_raw", 'train'), index=False, header=False)
