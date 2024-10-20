class TemporalBlock(nn.Module):
    def __init__(self, input_channels, output_channels, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        # First convolution layer with correct padding
        self.conv1 = nn.Conv1d(input_channels, output_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.conv2 = nn.Conv1d(output_channels, output_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout)
        
        # Skip connection (if input and output channels are different)
        self.downsample = nn.Conv1d(input_channels, output_channels, 1) if input_channels != output_channels else None
        
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        # First convolution + ReLU + Dropout
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        
        # Second convolution + ReLU + Dropout
        out = self.conv2(out)
        out = self.relu(out)
        out = self.dropout(out)

        # Residual connection (skip connection)
        res = x if self.downsample is None else self.downsample(x)
        
        # Ensure residuals and outputs match in size
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
            
            # Padding is adjusted to preserve sequence length
            padding = (kernel_size - 1) * dilation_size
            
            layers += [TemporalBlock(in_channels, out_channels, kernel_size, stride=1, 
                                     dilation=dilation_size, padding=padding, dropout=dropout)]

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], 1)  # Fully connected layer for final prediction

    def forward(self, x):
        # Input shape: (batch_size, num_features, sequence_length)
        out = self.network(x)  # Temporal blocks process the sequence
        out = out[:, :, -1]  # Take the output of the last time step in the sequence
        out = self.fc(out)  # Fully connected layer
        return out.squeeze()  # Squeeze to get the correct output shape
