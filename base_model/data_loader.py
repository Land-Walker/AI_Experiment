import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import config  # PRESERVED: Your config module import

# Download yf data
def load_yf_data(ticker_name="^GSPC", start_date="2024-01-01"):
    """ 
    Get data from yfinance 
    
    Downloads financial time series data from Yahoo Finance for a given ticker.
    Default downloads S&P 500 index data from 2024 onwards.
    
    Args:
        ticker_name: Stock ticker symbol (e.g., "AAPL", "^GSPC" for S&P 500)
        start_date: Start date for data download in YYYY-MM-DD format
    
    Returns:
        pd.DataFrame: DataFrame with Date, Open, High, Low, Close, Volume columns
    """
    # Download data using yfinance library - handles API calls to Yahoo Finance
    data = yf.download(ticker_name, start=start_date)
    
    # Convert to standard DataFrame and ensure Date is a column (not index)
    yfDf = pd.DataFrame(data)
    yfDf.reset_index(inplace=True)  # Moves Date from index to regular column
    return yfDf

def show_yf_data(dataset, num_samples=20):
    """
    Show data from yfinance 
    
    Creates a simple line plot of the closing price over time for visual inspection.
    
    Args:
        dataset: DataFrame with financial data
        num_samples: Currently unused parameter (kept for compatibility)
    """
    # Use pandas built-in plotting: x-axis is Date, y-axis is closing price
    dataset.plot(x='Date', y='Close')  
    plt.title('Stock Price Data')
    plt.show()

# Load & Preprocess Dataset for training
class TimeSeriesDataset(Dataset):
    """
    Custom dataset for time series diffusion models
    
    Handles loading, preprocessing, normalization, and sequence creation for time series data.
    Supports multiple data sources: yfinance DataFrames, CSV files, or numpy arrays.
    """
    def __init__(self, data_path=None, data=None, yf_dataframe=None, use_columns=None, sequence_length=None, normalize=True):
        """
        Initialize the dataset with flexible data source options
        
        Args:
            data_path: Path to CSV file with time series data
            data: Numpy array or tensor with time series data (alternative to data_path)
            yf_dataframe: DataFrame from yfinance (downloaded financial data)
            use_columns: List of columns to use from yfinance data (e.g., ['Open', 'High', 'Low', 'Close'])
            sequence_length: Length of sequences to create. If None, uses full length
            normalize: Whether to normalize data to [-1, 1] range (required for diffusion models)
        """
        if yf_dataframe is not None:
            # Handle yfinance DataFrame - need to filter out non-numeric columns
            if use_columns is None:
                # Auto-detect numeric columns by excluding common date column names
                all_cols = yf_dataframe.columns.tolist()
                exclude_cols = ['Date', 'date', 'DateTime', 'datetime', 'Datetime', 'timestamp', 'Timestamp']
                potential_cols = [col for col in all_cols if col not in exclude_cols]

                # Further filter to only numeric data types (float, int)
                numeric_data = yf_dataframe[potential_cols].select_dtypes(include=[np.number])
                use_columns = numeric_data.columns.tolist()

            # Extract specified columns and convert to float32 for PyTorch compatibility
            self.data = yf_dataframe[use_columns].values.astype(np.float32)
            self.column_names = use_columns

        elif data_path is not None:
            # Load from CSV file
            df = pd.read_csv(data_path)
            self.data = df.values.astype(np.float32)
            self.column_names = df.columns.tolist()

        elif data is not None:
            # Use provided numpy array or tensor directly
            self.data = np.array(data, dtype=np.float32)
            if len(self.data.shape) > 1:
                self.column_names = [f"Feature_{i}" for i in range(self.data.shape[1])]
            else:
                self.column_names = ["Feature_0"]

        else:
            # Generate synthetic data for demo/testing purposes
            self.data = self._generate_synthetic_data()
            self.column_names = ["Synthetic_1", "Synthetic_2", "Synthetic_3"]

        # Calculate dimensions automatically from data
        self.total_length = self.data.shape[0]  # Number of time steps
        self.input_dim = self.data.shape[1] if len(self.data.shape) > 1 else 1  # Number of features

        # Ensure data is 2D: [time_steps, features]
        if len(self.data.shape) == 1:
            self.data = self.data.reshape(-1, 1)

        # Set up sequence windowing
        if sequence_length is None:
            # Use entire dataset as one sequence
            self.sequence_length = self.total_length
            self.num_sequences = 1
        else:
            # Create sliding windows of specified length
            self.sequence_length = min(sequence_length, self.total_length)
            # Calculate how many sequences we can create with sliding window
            self.num_sequences = max(1, self.total_length - self.sequence_length + 1)

        # Normalize to [-1, 1] range (critical requirement for diffusion models)
        if normalize:
            # Compute min/max for each feature separately for proper scaling
            self.data_min = self.data.min(axis=0, keepdims=True)
            self.data_max = self.data.max(axis=0, keepdims=True)
            
            # Reshape for proper broadcasting with data
            self.data_min = self.data_min.squeeze(0)  # Shape: [features]
            self.data_max = self.data_max.squeeze(0)  # Shape: [features]

            # Apply normalization: scale to [0,1] then shift to [-1,1]
            self.data = 2 * (self.data - self.data_min) / (self.data_max - self.data_min) - 1
        else:
            self.data_min = None
            self.data_max = None

    def _generate_synthetic_data(self, length=1000, dim=3):
        """
        Generate synthetic time series for demo purposes
        
        Creates realistic-looking data with multiple sinusoidal components and noise.
        Useful for testing the pipeline without real financial data.
        
        Args:
            length: Number of time steps to generate
            dim: Number of features/dimensions
            
        Returns:
            np.array: Synthetic time series data with shape [length, dim]
        """
        # Create time axis
        t = np.linspace(0, 10, length)
        
        # Generate multiple sine waves with different frequencies + noise
        data = np.column_stack([
            np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(length),  # Slow oscillation
            np.sin(2 * np.pi * 2 * t) + 0.1 * np.random.randn(length),    # Fast oscillation
            0.5 * np.sin(2 * np.pi * 0.8 * t) + 0.05 * np.random.randn(length)  # Medium oscillation
        ])
        return data.astype(np.float32)

    def __len__(self):
        """Return number of sequences available in dataset"""
        return self.num_sequences

    def __getitem__(self, idx):
        """
        Get a single sequence by index
        
        Args:
            idx: Index of sequence to retrieve
            
        Returns:
            torch.Tensor: Sequence with shape [features, time] for Conv1d compatibility
        """
        if self.num_sequences == 1:
            # Return full sequence if using entire dataset
            sequence = self.data
        else:
            # Return sliding window sequence
            start_idx = idx
            end_idx = start_idx + self.sequence_length
            sequence = self.data[start_idx:end_idx]

        # Convert to PyTorch tensor
        tensor = torch.FloatTensor(sequence)

        # Transpose to [features, time] format required by Conv1d layers
        if len(tensor.shape) == 2 and tensor.shape[1] == self.input_dim:
            tensor = tensor.transpose(0, 1)  # [time, features] → [features, time]

        return tensor

    def denormalize(self, normalized_data):
        """
        Convert normalized data back to original scale
        
        Reverses the [-1,1] normalization to get back original data values.
        Important for visualization and evaluation of generated samples.
        
        Args:
            normalized_data: Data in [-1, 1] range
            
        Returns:
            np.array: Data in original scale
        """
        if self.data_min is not None and self.data_max is not None:
            # Handle different input formats
            if isinstance(normalized_data, torch.Tensor):
                normalized_data = normalized_data.detach().cpu().numpy()

            # Reshape min/max for proper broadcasting
            data_min = self.data_min[:, np.newaxis]  # [features, 1]
            data_max = self.data_max[:, np.newaxis]  # [features, 1]

            # Reverse normalization: (data + 1) / 2 * (max - min) + min
            return (normalized_data + 1) / 2 * (data_max - data_min) + data_min
        return normalized_data

def load_timeseries_dataset(data_path=None, data=None, yf_dataframe=None, use_columns=None, sequence_length=None):
    """
    Load and prepare time series dataset for diffusion model training
    
    Main function for creating dataset and dataloader. Automatically updates global
    configuration variables and returns everything needed for training.

    Args:
        data_path: Path to CSV file
        data: Numpy array with time series data
        yf_dataframe: DataFrame from yfinance
        use_columns: List of columns to use from yfinance data
        sequence_length: Length of sequences (None for full length)

    Returns:
        tuple: (dataset, dataloader, data_info) containing:
            - dataset: TimeSeriesDataset object
            - dataloader: PyTorch DataLoader for batch training
            - data_info: Dictionary with dataset metadata
    """
    # Create the dataset object
    dataset = TimeSeriesDataset(
        data_path=data_path, 
        data=data, 
        yf_dataframe=yf_dataframe, 
        use_columns=use_columns, 
        sequence_length=sequence_length
    )

    # PRESERVED: Update global config variables for other modules to use
    config.SEQUENCE_LENGTH = dataset.sequence_length
    config.INPUT_DIM = dataset.input_dim

    # Create PyTorch DataLoader for efficient batch processing
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,  # PRESERVED: Use config module
        shuffle=True,                  # Randomly shuffle data each epoch
        drop_last=True                 # Drop incomplete final batch
    )

    # Create information dictionary about the dataset
    data_range = '[-1, 1] (normalized)' if dataset.data_min is not None else 'original'
    data_info = {
        'sequence_length': config.SEQUENCE_LENGTH,  # PRESERVED: Use config module
        'input_dim': config.INPUT_DIM,              # PRESERVED: Use config module
        'total_sequences': len(dataset),
        'batch_size': config.BATCH_SIZE,            # PRESERVED: Use config module
        'data_range': data_range,
        'columns': getattr(dataset, 'column_names', None)
    }

    return dataset, dataloader, data_info

def show_timeseries_sample(data_sample, dataset=None, title="Time Series Sample"):
    """
    Visualize time series data with proper handling of different tensor formats
    
    Creates separate plots for each feature/dimension in the time series.
    Handles various input shapes and automatically denormalizes if possible.

    Args:
        data_sample: Tensor with time series data (various possible shapes)
        dataset: Dataset object for denormalization and column names (optional)
        title: Title for the plots
    """
    # Ensure data is on CPU for plotting
    if torch.is_tensor(data_sample):
        data_sample = data_sample.detach().cpu()

    # Handle batch dimension
    if data_sample.dim() == 3:
        # Remove batch dimension: [batch, features, time] → [features, time]
        data_sample = data_sample[0]
        
        # Check if we need to transpose based on expected dimensions
        if data_sample.shape[1] != config.INPUT_DIM and data_sample.shape[0] == config.INPUT_DIM:  # PRESERVED
             # Currently [time, features], transpose to [features, time]
             data_sample = data_sample.transpose(0, 1)
        elif data_sample.shape[0] != config.INPUT_DIM and data_sample.shape[1] != config.INPUT_DIM:  # PRESERVED
             print(f"Warning: Unexpected sample shape {data_sample.shape} after taking batch[0]. Attempting transpose.")
             data_sample = data_sample.transpose(0, 1)

    elif data_sample.dim() == 2:
        # Ensure correct orientation: [features, time]
        if data_sample.shape[1] != config.INPUT_DIM and data_sample.shape[0] == config.INPUT_DIM:  # PRESERVED
             # Currently [time, features], transpose to [features, time]
             data_sample = data_sample.transpose(0, 1)
    else:
         raise ValueError(f"Expected data_sample to have 2 or 3 dimensions, but got {data_sample.dim()}")

    # Extract dimensions for plotting
    num_features = data_sample.shape[0]
    sequence_length = data_sample.shape[1]

    # Attempt denormalization if dataset supports it
    if dataset is not None and hasattr(dataset, 'denormalize'):
        try:
            # Add batch dimension for denormalization, then remove it
            data_sample_denorm = dataset.denormalize(data_sample.unsqueeze(0)).squeeze(0)
        except Exception as e:
            print(f"Warning: Could not denormalize data for plotting: {e}")
            data_sample_denorm = data_sample
    else:
        data_sample_denorm = data_sample

    # Create subplots for each feature
    fig, axes = plt.subplots(num_features, 1, figsize=(12, 3*num_features))
    if num_features == 1:
        axes = [axes]  # Ensure axes is always iterable

    # Plot each feature in its own subplot
    for i in range(num_features):
        axes[i].plot(data_sample_denorm[i])

        # Get meaningful column name or use generic name
        col_name = f'Dimension {i+1}'
        if dataset is not None and hasattr(dataset, 'column_names') and len(dataset.column_names) > i:
             col_name = dataset.column_names[i]

        axes[i].set_title(f'{title} - {col_name}')
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

# Example usage documentation preserved exactly as in your code
"""
Example usage:
if __name__ == "__main__":
    # Download yfinance data
    yf_df = load_yf_data()
    show_yf_data(yf_df)

    # Option 1: Use your yfDf with selected columns
    dataset, dataloader, info = load_timeseries_dataset(
        yf_dataframe=yf_df,  # Your existing variable
        use_columns=['Open', 'High', 'Low', 'Close'],  # Exclude Volume for cleaner data
        sequence_length=60  # 60-day sequences
    )

    # Option 2: Use all OHLCV data from your yfDf (auto-detects numeric columns)
    # dataset, dataloader, info = load_timeseries_dataset(yf_dataframe=yfDf, sequence_length=30)

    # Option 3: Just closing prices from your yfDf
    # dataset, dataloader, info = load_timeseries_dataset(yf_dataframe=yfDf, use_columns=['Close'], sequence_length=100)

    # Print dataset information
    print("Dataset Information:")
    for key, value in info.items():
        print(f"  {key}: {value}")

    # Show a sample
    sample_batch = next(iter(dataloader))
    print(f"\nSample batch shape: {sample_batch.shape}")
    show_timeseries_sample(sample_batch, dataset, "Stock Price Time Series")
"""