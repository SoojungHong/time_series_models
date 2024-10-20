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
        # Extract datetime and process as features
        self.dates = pd.to_datetime(df['datetime'])
        self.df = df
        
        # Preprocessing: handle datetime features
        self.df['hour'] = self.dates.dt.hour
        self.df['day_of_week'] = self.dates.dt.dayofweek
        self.df['month'] = self.dates.dt.month
        
        # Preprocessing: currency pair (categorical -> numerical)
        le = LabelEncoder()
        self.df['currency_pair_encoded'] = le.fit_transform(df['currency_pair'])
        
        # Features to use (you can add more datetime-derived features as needed)
        self.feature_columns = ['currency_pair_encoded', 'volume_last_minute', 
                                'volume_last_hour', 'mid_price', 'hour', 'day_of_week', 'month']
        
        # Scale the features (you could apply more sophisticated scaling if needed)
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.df[self.feature_columns])
        
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
# ['datetime', 'currency_pair', 'volume_last_minute', 'volume_last_hour', 'mid_price', 'volume_next_hour']

if __name__ == '__main__':
    # Example DataFrame for testing (you would load your actual dataset)
    data = {
        'datetime': ['2024-01-01 10:00:00', '2024-01-01 10:01:00', '2024-01-01 10:02:00'],
        'currency_pair': ['EURUSD', 'GBPUSD', 'EURCHF'],
        'volume_last_minute': [120.5, 110.0, 135.3],
        'volume_last_hour': [5000.0, 4800.0, 5200.0],
        'mid_price': [1.1050, 1.3075, 1.0812],
        'volume_next_hour': [6000.0, 5800.0, 6400.0]
    }
    fx_df = pd.DataFrame(data)
    
    # Create FXDataset
    dataset = FXDataset(fx_df)
    
    # Example access to dataset
    features, target = dataset[0]
    print(f"Features: {features}")
    print(f"Target: {target}")
