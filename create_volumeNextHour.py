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



import torch
import torch.nn as nn

class TemporalBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # First convolution layer
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Add the skip connection if input and output channels don't match
        self.downsample = nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else None
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Add the skip connection (residual connection)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)

class TCN(nn.Module):
    def __init__(self, num_inputs, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)
        
        for i in range(num_levels):
            dilation_size = 2 ** i  # Dilations grow exponentially (1, 2, 4, 8, ...)
            in_channels = num_inputs if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                     dilation=dilation_size, padding=(kernel_size - 1) * dilation_size, 
                                     dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)  # Fully connected layer for final prediction

    def forward(self, x):
        # Input shape: (batch_size, num_features, sequence_length)
        out = self.network(x)  # Temporal blocks process the sequence
        out = out[:, :, -1]  # Take the output of the last time step in the sequence
        out = self.fc(out)  # Fully connected layer
        return out.squeeze()  # Squeeze to get the correct output shape


from torch.utils.data import DataLoader

# Assume fx_df is your DataFrame loaded with the necessary data
sequence_length = 10  # Number of past time steps in the input sequence
fx_dataset = FXDataset(fx_df, sequence_length=sequence_length)

# Create DataLoader for batching
batch_size = 32
dataloader = DataLoader(fx_dataset, batch_size=batch_size, shuffle=True)


# Define model parameters
input_size = len(fx_dataset.feature_columns)  # Number of input features per time step
num_channels = [32, 32, 32]  # Number of channels for each temporal block
kernel_size = 2
dropout = 0.2

# Initialize the TCN model
model = TCN(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)

# Set up optimizer and loss function
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = nn.MSELoss()  # You can also use MAE depending on your needs

# Training loop
epochs = 50

for epoch in range(epochs):
    model.train()  # Set the model to training mode
    total_loss = 0
    
    for batch in dataloader:
        features, target = batch
        
        # Reshape features to (batch_size, num_features, sequence_length)
        features = features.permute(0, 2, 1)
        
        # Zero out gradients
        optimizer.zero_grad()
        
        # Forward pass
        predictions = model(features)
        
        # Compute loss
        loss = criterion(predictions, target)
        
        # Backward pass and optimization
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
    
    print(f"Epoch {epoch + 1}/{epochs}, Loss: {total_loss / len(dataloader)}")


