import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import os

class DataManager:
    def __init__(self):
        self.supported_formats = ['.txt', '.csv']
    
    def load_data(self, file_path):
        """Load data from various file formats"""
        file_ext = os.path.splitext(file_path.name)[-1].lower()
        
        if file_ext == '.csv':
            return self._load_csv(file_path)
        elif file_ext == '.txt':
            return self._load_txt(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_ext}. Supported formats: {self.supported_formats}")
    
    def _load_csv(self, file_path):
        """Load CSV file - expecting either a single text column or text,label columns"""
        df = pd.read_csv(file_path)
        
        # If only one column, assume it's text and add placeholder labels
        if len(df.columns) == 1:
            df.columns = ['text']
            df['label'] = -1  # Placeholder for unlabeled data
        elif len(df.columns) >= 2:
            # Use first two columns as text and label
            df = df.iloc[:, :2]
            df.columns = ['text', 'label']
        
        return df
    
    def _load_txt(self, file_path):
        """Load text file - each line becomes a data point"""
        with open(file_path, 'r', encoding='utf-8') as f:
            lines = [line.strip() for line in f.readlines() if line.strip()]
        
        df = pd.DataFrame({
            'text': lines,
            'label': -1  # Placeholder for unlabeled data
        })
        return df
    
    def split_data(self, df, test_size=0.3, random_state=42):
        """Split data into pool (for labeling) and test set"""
        if len(df) < 10:
            raise ValueError("Need at least 10 data points for splitting")
        
        # Filter out any rows with missing text
        df = df.dropna(subset=['text'])
        df = df[df['text'].str.strip().astype(bool)]
        
        # If we have some labeled data, use stratified split
        if (df['label'] != -1).any():
            labeled_mask = df['label'] != -1
            labeled_data = df[labeled_mask]
            unlabeled_data = df[~labeled_mask]
            
            if len(labeled_data) >= 10:
                train_df, test_df = train_test_split(
                    labeled_data, 
                    test_size=test_size, 
                    random_state=random_state,
                    stratify=labeled_data['label']
                )
                pool_df = pd.concat([train_df, unlabeled_data])
            else:
                # Not enough labeled data for stratified split
                train_df, test_df = train_test_split(
                    df, 
                    test_size=test_size, 
                    random_state=random_state
                )
                pool_df = train_df
        else:
            # No labels, do simple split
            pool_df, test_df = train_test_split(
                df, 
                test_size=test_size, 
                random_state=random_state
            )
        
        return pool_df.reset_index(drop=True), test_df.reset_index(drop=True)
    
    def get_labeling_sample(self, pool_df, sample_size=20):
        """Get a random sample for user labeling"""
        if len(pool_df) <= sample_size:
            return pool_df.copy()
        
        return pool_df.sample(n=sample_size, random_state=42).copy()