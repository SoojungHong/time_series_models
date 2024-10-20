
import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FXDataset(Dataset):
    def __init__(self, df, target_column='volume_next_hour'):
        """
        Initializes the FXDataset.
        
        Parameters:
        - df: Pandas DataFrame containing the features and target.
        - target_column: The name of the target column (volume in the next hour).
        """
        # Extract and process datetime features
        self.dates = pd.to_datetime(df['datetime'])
        self.df = df.copy()  # To avoid modifying the original dataframe
        
        # Extract useful datetime components as features
        self.df['hour'] = self.dates.dt.hour
        self.df['day_of_week'] = self.dates.dt.dayofweek
        self.df['month'] = self.dates.dt.month
        
        # Handle currency pair: label encoding or one-hot encoding
        le = LabelEncoder()
        self.df['currency_pair_encoded'] = le.fit_transform(df['currency_pair'])
        
        # Select features for training
        self.feature_columns = [
            'currency_pair_encoded', 'volume_last_minute', 'volume_last_hour', 
            'mid_price', 'spread', 'hour', 'day_of_week', 'month'
        ]
        
        # Standardize numerical features
        scaler = StandardScaler()
        self.features = scaler.fit_transform(self.df[self.feature_columns])
        
        # Target: volume in next hour
        self.target = df[target_column].values
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        """
        Returns a single sample from the dataset.
        
        Parameters:
        - idx: Index of the sample
        
        Returns:
        - A tuple (features, target) where features are scaled inputs and target is the next hour's volume.
        """
        features = torch.tensor(self.features[idx], dtype=torch.float32)
        target = torch.tensor(self.target[idx], dtype=torch.float32)
        return features, target

# Example usage:
# Assuming you have a pandas DataFrame `fx_df` with the following columns:
# ['datetime', 'currency_pair', 'volume_last_minute', 'volume_last_hour', 'mid_price', 'spread', 'volume_next_hour']

if __name__ == '__main__':
    # Sample DataFrame for testing (replace this with your actual data)
    data = {
        'datetime': ['2024-01-01 10:00:00', '2024-01-01 10:01:00', '2024-01-01 10:02:00'],
        'currency_pair': ['EURUSD', 'GBPUSD', 'EURCHF'],
        'volume_last_minute': [120.5, 110.0, 135.3],
        'volume_last_hour': [5000.0, 4800.0, 5200.0],
        'mid_price': [1.1050, 1.3075, 1.0812],
        'spread': [0.0002, 0.0003, 0.0004],
        'volume_next_hour': [6000.0, 5800.0, 6400.0]
    }
    fx_df = pd.DataFrame(data)
    
    # Create FXDataset
    dataset = FXDataset(fx_df)
    
    # Example access to dataset
    features, target = dataset[0]
    print(f"Features: {features}")
    print(f"Target: {target}")
