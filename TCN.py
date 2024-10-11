import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tcn import TCN, tcn_full_summary
from sklearn.metrics import mean_absolute_percentage_error

class TCNModel:
    def __init__(self, input_shape, nb_filters=64, kernel_size=3, dilations=[1, 2, 4, 8], dropout_rate=0.2):
        """
        Initialize the TCN model with the given parameters.

        Parameters:
        - input_shape: The shape of the input data (timesteps, features)
        - nb_filters: Number of filters in the convolutional layers
        - kernel_size: Size of the convolution kernel
        - dilations: List of dilation rates for the dilated convolutions
        - dropout_rate: Dropout rate for regularization
        """
        self.input_shape = input_shape
        self.nb_filters = nb_filters
        self.kernel_size = kernel_size
        self.dilations = dilations
        self.dropout_rate = dropout_rate
        self.model = self.build_model()
    
    def build_model(self):
        """
        Build and compile the TCN model.
        Returns:
        - A compiled TCN model.
        """
        model = Sequential()
        
        # Add a TCN layer
        model.add(TCN(nb_filters=self.nb_filters,
                      kernel_size=self.kernel_size,
                      dilations=self.dilations,
                      input_shape=self.input_shape,
                      dropout_rate=self.dropout_rate,
                      return_sequences=False))
        
        # Add Dense layers
        model.add(Dense(64, activation='relu'))  # A Dense layer for further processing
        model.add(Dense(1))                      # Output layer for regression (predicting a single value)
        
        # Compile the model
        model.compile(optimizer='adam', loss='mse', metrics=['mae'])
        
        # Print a summary of the model
        tcn_full_summary(model)
        
        return model
    
    def fit(self, X_train, y_train, epochs=20, batch_size=32, validation_split=0.2):
        """
        Train the TCN model.
        
        Parameters:
        - X_train: Training input data
        - y_train: Training target data
        - epochs: Number of training epochs
        - batch_size: Batch size for training
        - validation_split: Fraction of the training data to be used as validation data
        """
        history = self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=validation_split)
        return history

    def evaluate(self, X_test, y_test):
        """
        Evaluate the TCN model on the test set.
        
        Parameters:
        - X_test: Test input data
        - y_test: Test target data
        
        Returns:
        - Loss and MAE (Mean Absolute Error) on the test data.
        """
        return self.model.evaluate(X_test, y_test)

    def predict(self, X):
        """
        Make predictions using the trained TCN model.
        
        Parameters:
        - X: Input data for which predictions are to be made
        
        Returns:
        - Predicted values.
        """
        return self.model.predict(X)

    def calculate_mape(self, y_true, y_pred):
        """
        Calculate the Mean Absolute Percentage Error (MAPE) between true and predicted values.
        
        Parameters:
        - y_true: Actual target values
        - y_pred: Predicted values
        
        Returns:
        - MAPE score.
        """
        return mean_absolute_percentage_error(y_true, y_pred)

# Example usage:
if __name__ == '__main__':
    # Example input data
    input_shape = (1, 3)  # 1 timestep and 3 features (e.g., volume_last_minute, volume_last_hour, mid_price)
    
    # Initialize the TCN model
    tcn_model = TCNModel(input_shape=input_shape, nb_filters=64, kernel_size=3, dilations=[1, 2, 4, 8], dropout_rate=0.2)
    
    # Example training and test data (replace these with actual data)
    # X_train_scaled, y_train, X_test_scaled, y_test would be your actual datasets
    X_train_scaled = np.random.rand(1000, 1, 3)  # Example: 1000 samples, 1 timestep, 3 features
    y_train = np.random.rand(1000)
    X_test_scaled = np.random.rand(200, 1, 3)    # Example: 200 test samples
    y_test = np.random.rand(200)
    
    # Train the model
    tcn_model.fit(X_train_scaled, y_train, epochs=20, batch_size=32)
    
    # Evaluate the model
    tcn_model.evaluate(X_test_scaled, y_test)
    
    # Make predictions
    predictions = tcn_model.predict(X_test_scaled)
    
    # Calculate MAPE
    mape = tcn_model.calculate_mape(y_test, predictions)
    print(f'MAPE: {mape:.4f}')
