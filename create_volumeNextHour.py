import torch
from torch.utils.data import Dataset
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder

class FXDataset(Dataset):
    def __init__(self, df, sequence_length=10):
        """
        Initializes the FXDataset.
        
        Parameters:
        - df: Pandas DataFrame containing the features.
        - sequence_length: How many past time steps to include in each sequence.
        """
        self.sequence_length = sequence_length
        self.dates = pd.to_datetime(df['datetime'])
        self.df = df.copy()
        
        # Extract useful datetime components as features
        self.df['hour'] = self.dates.dt.hour
        self.df['day_of_week'] = self.dates.dt.dayofweek
        self.df['month'] = self.dates.dt.month
        
        # Handle currency pair: label encoding
        le = LabelEncoder()
        self.df['currency_pair_encoded'] = le.fit_transform(df['currency_pair'])
        
        # Create the target (volume_next_hour) by shifting 'volume_last_hour' column
        self.df['volume_next_hour'] = self.df['volume_last_hour'].shift(-1)
        
        # Remove the last row since it won't have a valid target after shifting
        self.df = self.df[:-1]
        
        # Select features for training
        self.feature_columns = [
            'currency_pair_encoded', 'volume_last_minute', 'volume_last_hour', 
            'mid_price', 'spread', 'hour', 'day_of_week', 'month'
        ]
        
        # Standardize numerical features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.df[self.feature_columns])
        
        # Target: volume in next hour (after the shift)
        self.target = self.df['volume_next_hour'].values
    
    def __len__(self):
        return len(self.df) - self.sequence_length

    def __getitem__(self, idx):
        """
        Returns a sequence of feature vectors and the target (volume_next_hour).
        
        Parameters:
        - idx: Index of the starting point of the sequence
        
        Returns:
        - A tuple (features_sequence, target)
        """
        # Get a sequence of features from idx to idx + sequence_length
        feature_sequence = self.features[idx:idx + self.sequence_length]
        
        # The target is volume_next_hour at the index idx + sequence_length
        target = self.target[idx + self.sequence_length]
        
        # Convert to tensors
        feature_sequence = torch.tensor(feature_sequence, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        
        return feature_sequence, target



# Example DataFrame (replace with your actual data)
data = {
    'datetime': ['2024-01-01 10:00:00', '2024-01-01 10:01:00', '2024-01-01 10:02:00', 
                 '2024-01-01 10:03:00', '2024-01-01 10:04:00', '2024-01-01 10:05:00'],
    'currency_pair': ['EURUSD', 'GBPUSD', 'EURCHF', 'EURUSD', 'GBPUSD', 'EURCHF'],
    'volume_last_minute': [120.5, 110.0, 135.3, 125.4, 105.6, 130.0],
    'volume_last_hour': [5000.0, 4800.0, 5200.0, 5100.0, 4900.0, 5300.0],
    'mid_price': [1.1050, 1.3075, 1.0812, 1.1060, 1.3080, 1.0820],
    'spread': [0.0002, 0.0003, 0.0004, 0.0002, 0.0003, 0.0004]
}
fx_df = pd.DataFrame(data)

# Create the dataset
sequence_length = 3  # Number of time steps in the input sequence
fx_dataset = FXDataset(fx_df, sequence_length=sequence_length)

# Example access to the dataset
features, target = fx_dataset[0]
print(f"Features: {features}")
print(f"Target (volume_next_hour): {target}")



from torch.utils.data import DataLoader

# Define parameters
sequence_length = 10
batch_size = 32
epochs = 50
learning_rate = 0.001

# Initialize the dataset and dataloader
fx_dataset = FXDataset(fx_df, sequence_length=sequence_length)
dataloader = DataLoader(fx_dataset, batch_size=batch_size, shuffle=True)

# Define the TCN model
input_size = len(fx_dataset.feature_columns)  # Number of input features
output_size = 1  # Predicting next hour's volume
num_channels = [32, 32, 32]  # Number of channels in the TCN layers

model = TCN(input_size, num_channels)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
criterion = nn.MSELoss()  # You can also use MAE for Mean Absolute Error

# Training loop
model.train()
for epoch in range(epochs):
    total_loss = 0
    for features, target in dataloader:
        features = features.permute(0, 2, 1)  # Reshape to (batch_size, num_features, sequence_length)
        
        optimizer.zero_grad()
        predictions = model(features)
        loss = criterion(predictions, target)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss / len(dataloader)}")


