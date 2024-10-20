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


from torch.utils.data import DataLoader

# Initialize dataset and dataloader
dataset = FXDataset(fx_df)  # Use your actual data here
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

# Now the dataloader can be used in training
for features, target in dataloader:
    # Your training loop here
    pass



import torch
import torch.nn as nn
import torch.optim as optim

class TemporalBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.net = nn.Sequential(
            self.conv1,
            self.relu,
            self.dropout,
            self.conv2,
            self.relu,
            self.dropout
        )
        self.downsample = nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)  # Output layer to predict next hour volume

    def forward(self, x):
        # x shape is (batch_size, num_features, sequence_length)
        out = self.network(x)
        out = out[:, :, -1]  # Take the last output in the sequence
        out = self.fc(out)
        return out.squeeze()  # Remove extra dimensions


class FXDataset(Dataset):
    def __init__(self, df, sequence_length=10, target_column='volume_next_hour'):
        """
        Initializes the FXDataset.
        
        Parameters:
        - df: Pandas DataFrame containing the features and target.
        - sequence_length: How many past time steps to include in each sequence.
        - target_column: The name of the target column (volume in the next hour).
        """
        self.sequence_length = sequence_length
        self.dates = pd.to_datetime(df['datetime'])
        self.df = df.copy()
        
        # Datetime feature engineering
        self.df['hour'] = self.dates.dt.hour
        self.df['day_of_week'] = self.dates.dt.dayofweek
        self.df['month'] = self.dates.dt.month
        
        # Encode the currency pair
        le = LabelEncoder()
        self.df['currency_pair_encoded'] = le.fit_transform(df['currency_pair'])
        
        # Feature columns
        self.feature_columns = ['currency_pair_encoded', 'volume_last_minute', 'volume_last_hour', 
                                'mid_price', 'spread', 'hour', 'day_of_week', 'month']
        
        # Standardize the features
        self.scaler = StandardScaler()
        self.features = self.scaler.fit_transform(self.df[self.feature_columns])
        
        # Target: next hour's volume
        self.target = df[target_column].values
    
    def __len__(self):
        return len(self.df) - self.sequence_length

    def __getitem__(self, idx):
        """
        Returns a sequence of feature vectors and a target value.
        
        Parameters:
        - idx: Index of the starting point of the sequence
        
        Returns:
        - A tuple (features_sequence, target)
        """
        feature_sequence = self.features[idx:idx + self.sequence_length]
        target = self.target[idx + self.sequence_length]  # Target corresponds to the hour after the sequence
        feature_sequence = torch.tensor(feature_sequence, dtype=torch.float32)
        target = torch.tensor(target, dtype=torch.float32)
        return feature_sequence, target


from torch.utils.data import DataLoader

# Set parameters
sequence_length = 10  # How many past time steps to use for each sequence
batch_size = 32
epochs = 50
learning_rate = 0.001

# Initialize dataset and dataloader
dataset = FXDataset(fx_df, sequence_length=sequence_length)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

# Define the TCN model
input_size = len(dataset.feature_columns)
output_size = 1  # Predicting next hour's volume
num_channels = [32, 32, 32]  # Number of channels for each TCN layer

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


