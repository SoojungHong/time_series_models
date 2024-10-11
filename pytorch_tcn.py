import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_percentage_error
import numpy as np

# Step 1: Create a PyTorch Dataset for the FX time series
class FXDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]

# Step 2: Define the Temporal Convolutional Network (TCN)
class TCNBlock(nn.Module):
    def __init__(self, input_size, output_size, kernel_size=2, dilation=1):
        super(TCNBlock, self).__init__()
        self.conv1 = nn.Conv1d(input_size, output_size, kernel_size, dilation=dilation, padding=(kernel_size-1)*dilation)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.conv2 = nn.Conv1d(output_size, output_size, kernel_size, dilation=dilation, padding=(kernel_size-1)*dilation)
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.relu(out)
        return out

class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2, dropout=0.2):
        super(TCN, self).__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_size if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            layers += [TCNBlock(in_channels, out_channels, kernel_size, dilation=dilation_size)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)

    def forward(self, x):
        x = self.network(x)
        x = x[:, :, -1]  # Take the output from the last timestep
        x = self.fc(x)
        return x

# Step 3: Train the TCN model
def train_model(model, dataloader, criterion, optimizer, num_epochs):
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for inputs, targets in dataloader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.squeeze(), targets)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * inputs.size(0)
        epoch_loss = running_loss / len(dataloader.dataset)
        print(f'Epoch {epoch+1}/{num_epochs}, Loss: {epoch_loss:.4f}')

# Step 4: Evaluation
def evaluate_model(model, dataloader):
    model.eval()
    predictions, actuals = [], []
    with torch.no_grad():
        for inputs, targets in dataloader:
            outputs = model(inputs)
            predictions.append(outputs.squeeze().numpy())
            actuals.append(targets.numpy())
    
    predictions = np.concatenate(predictions)
    actuals = np.concatenate(actuals)
    
    mape = mean_absolute_percentage_error(actuals, predictions)
    print(f'MAPE: {mape:.4f}')
    return mape

# Step 5: Main function to set up data, train, and evaluate
if __name__ == '__main__':
    # Example dataset (Replace with actual data)
    # X_train_scaled, X_test_scaled: (samples, timesteps, features), where timesteps is 1 for this use case
    # y_train, y_test: Target data

    np.random.seed(42)
    torch.manual_seed(42)
    
    # Simulating some random data for the example
    X_train_scaled = np.random.rand(1000, 1, 3)  # 1000 samples, 1 timestep, 3 features
    y_train = np.random.rand(1000)
    X_test_scaled = np.random.rand(200, 1, 3)   # 200 test samples
    y_test = np.random.rand(200)

    # Convert data to PyTorch Datasets
    train_dataset = FXDataset(X_train_scaled, y_train)
    test_dataset = FXDataset(X_test_scaled, y_test)

    # Create DataLoader
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    # Initialize TCN model
    input_size = X_train_scaled.shape[2]  # Number of features (3 in this case)
    num_channels = [64, 64, 64]  # Number of filters for each layer
    model = TCN(input_size=input_size, num_channels=num_channels)

    # Loss function and optimizer
    criterion = nn.MSELoss()  # Using MSE since it's a regression problem
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    train_model(model, train_loader, criterion, optimizer, num_epochs=20)

    # Evaluate the model
    evaluate_model(model, test_loader)
