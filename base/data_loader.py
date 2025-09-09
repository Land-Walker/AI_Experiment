import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf

# Download yf data
def load_yf_data(ticker_name="^GSPC", start_date="2024-01-01"):
  """ Get data from yfinance """
  data = yf.download(ticker_name, start=start_date)
  yfDf = pd.DataFrame(data)
  yfDf.reset_index(inplace=True)
  return yfDf

def show_yf_data(dataset, num_samples=20):
  """" Show data from yfinance """
  dataset.plot(x='Date', y='Close') # Use dataset.plot() for plotting
  plt.title('Stock Price Data')
  plt.show()

# Auto-calculated based on your data (These will be set when the dataset is created)
SEQUENCE_LENGTH = None  # Will be set automatically
INPUT_DIM = None        # Will be set automatically
BATCH_SIZE = 128

# Load & Preprocess Dataset for training
class TimeSeriesDataset(Dataset):
    """
    Custom dataset for time series diffusion models
    """
    def __init__(self, data_path=None, data=None, yf_dataframe=None, use_columns=None, sequence_length=None, normalize=True):
        """
        Args:
            data_path: Path to CSV file with time series data
            data: Numpy array or tensor with time series data (alternative to data_path)
            yf_dataframe: DataFrame from yfinance (your yfDf variable)
            use_columns: List of columns to use from yfinance data
            sequence_length: Length of sequences to create. If None, uses full length
            normalize: Whether to normalize data to [-1, 1] range
        """
        if yf_dataframe is not None:
            # Handle yfinance DataFrame - filter out Date/timestamp columns
            if use_columns is None:
                # Get all columns except Date/datetime columns
                all_cols = yf_dataframe.columns.tolist()
                # Explicitly exclude common date column names
                exclude_cols = ['Date', 'date', 'DateTime', 'datetime', 'Datetime', 'timestamp', 'Timestamp']
                potential_cols = [col for col in all_cols if col not in exclude_cols]

                # Further filter to only numeric types
                numeric_data = yf_dataframe[potential_cols].select_dtypes(include=[np.number])
                use_columns = numeric_data.columns.tolist()

            # Extract only the specified numeric columns
            self.data = yf_dataframe[use_columns].values.astype(np.float32)
            self.column_names = use_columns

        elif data_path is not None:
            # Load from CSV
            df = pd.read_csv(data_path)
            self.data = df.values.astype(np.float32)
            self.column_names = df.columns.tolist()

        elif data is not None:
            # Use provided data
            self.data = np.array(data, dtype=np.float32)
            if len(self.data.shape) > 1:
                self.column_names = [f"Feature_{i}" for i in range(self.data.shape[1])]
            else:
                self.column_names = ["Feature_0"]

        else:
            # Generate synthetic data for demo
            self.data = self._generate_synthetic_data()
            self.column_names = ["Synthetic_1", "Synthetic_2", "Synthetic_3"]

        # Auto-calculate dimensions
        self.total_length = self.data.shape[0]
        self.input_dim = self.data.shape[1] if len(self.data.shape) > 1 else 1

        if len(self.data.shape) == 1:
            self.data = self.data.reshape(-1, 1)

        # Set sequence length
        if sequence_length is None:
            self.sequence_length = self.total_length
            self.num_sequences = 1
        else:
            self.sequence_length = min(sequence_length, self.total_length)
            self.num_sequences = max(1, self.total_length - self.sequence_length + 1)

        # Normalize to [-1, 1] range (required for diffusion models)
        if normalize:
            self.data_min = self.data.min(axis=0, keepdims=True)
            self.data_max = self.data.max(axis=0, keepdims=True)
            # Ensure data_min/max have correct shape for broadcasting with [time, features] data
            self.data_min = self.data_min.squeeze(0) # Shape becomes [features]
            self.data_max = self.data_max.squeeze(0) # Shape becomes [features]

            self.data = 2 * (self.data - self.data_min) / (self.data_max - self.data_min) - 1
        else:
            self.data_min = None
            self.data_max = None

    def _generate_synthetic_data(self, length=1000, dim=3):
        """Generate synthetic time series for demo purposes"""
        t = np.linspace(0, 10, length)
        # Multiple sinusoidal components with noise
        data = np.column_stack([
            np.sin(2 * np.pi * 0.5 * t) + 0.1 * np.random.randn(length),  # Slow oscillation
            np.sin(2 * np.pi * 2 * t) + 0.1 * np.random.randn(length),    # Fast oscillation
            0.5 * np.sin(2 * np.pi * 0.8 * t) + 0.05 * np.random.randn(length)  # Medium oscillation
        ])
        return data.astype(np.float32)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        if self.num_sequences == 1:
            # Return full sequence
            sequence = self.data
        else:
            # Return sliding window
            start_idx = idx
            end_idx = start_idx + self.sequence_length
            sequence = self.data[start_idx:end_idx]

        # Convert to tensor and ensure correct shape: [features, time] for Conv1d
        tensor = torch.FloatTensor(sequence)

        # Transpose to [features, time] if currently [time, features]
        if len(tensor.shape) == 2 and tensor.shape[1] == self.input_dim:
            tensor = tensor.transpose(0, 1)  # [time, features] -> [features, time]

        return tensor

    def denormalize(self, normalized_data):
        """Convert normalized data back to original scale"""
        if self.data_min is not None and self.data_max is not None:
            # Handle different input shapes: [features, time] or [batch, features, time]
            if isinstance(normalized_data, torch.Tensor):
                normalized_data = normalized_data.detach().cpu().numpy()

            # Ensure data_min/max have correct shape for broadcasting with [features, time]
            # They should be [features, 1]
            data_min = self.data_min[:, np.newaxis]
            data_max = self.data_max[:, np.newaxis]

            return (normalized_data + 1) / 2 * (data_max - data_min) + data_min
        return normalized_data

def load_timeseries_dataset(data_path=None, data=None, yf_dataframe=None, use_columns=None, sequence_length=None):
    """
    Load and prepare time series dataset for diffusion model

    Args:
        data_path: Path to CSV file
        data: Numpy array with time series data
        yf_dataframe: DataFrame from yfinance (your yfDf variable)
        use_columns: List of columns to use from yfinance data
        sequence_length: Length of sequences (None for full length)

    Returns:
        dataset: TimeSeriesDataset object
        dataloader: DataLoader object
        data_info: Dictionary with dataset information
    """
    # Create dataset
    dataset = TimeSeriesDataset(data_path=data_path, data=data, yf_dataframe=yf_dataframe, use_columns=use_columns, sequence_length=sequence_length)

    # Auto-calculate global dimensions
    global SEQUENCE_LENGTH, INPUT_DIM
    SEQUENCE_LENGTH = dataset.sequence_length
    INPUT_DIM = dataset.input_dim

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    # Return dataset info
    data_range = '[-1, 1] (normalized)' if dataset.data_min is not None else 'original'
    data_info = {
        'sequence_length': SEQUENCE_LENGTH,
        'input_dim': INPUT_DIM,
        'total_sequences': len(dataset),
        'batch_size': BATCH_SIZE,
        'data_range': data_range,
        'columns': getattr(dataset, 'column_names', None)
    }

    return dataset, dataloader, data_info

def show_timeseries_sample(data_sample, dataset=None, title="Time Series Sample"):
    """
    Visualize time series data (equivalent to show_tensor_image for images)

    Args:
        data_sample: Tensor of shape (sequence_length, input_dim) or (batch_size, sequence_length, input_dim)
                     or (batch_size, input_dim, sequence_length) or (input_dim, sequence_length)
        dataset: Dataset object for denormalization (optional)
        title: Plot title
    """
    # Handle batch dimension and ensure shape is [features, time] for plotting
    if torch.is_tensor(data_sample):
        data_sample = data_sample.detach().cpu()

    if data_sample.dim() == 3:
        # Assume shape is [batch_size, features, time] or [batch_size, time, features]
        # Take the first sample and ensure it's [features, time]
        data_sample = data_sample[0]
        if data_sample.shape[1] != INPUT_DIM and data_sample.shape[0] == INPUT_DIM:
             # It's [time, features], transpose to [features, time]
             data_sample = data_sample.transpose(0, 1)
        elif data_sample.shape[0] != INPUT_DIM and data_sample.shape[1] != INPUT_DIM:
             print(f"Warning: Unexpected sample shape {data_sample.shape} after taking batch[0]. Attempting transpose.")
             data_sample = data_sample.transpose(0, 1) # Hope it becomes [features, time]

    elif data_sample.dim() == 2:
        # Assume shape is [features, time] or [time, features]
        # Ensure it's [features, time]
        if data_sample.shape[1] != INPUT_DIM and data_sample.shape[0] == INPUT_DIM:
             # It's [time, features], transpose to [features, time]
             data_sample = data_sample.transpose(0, 1)

    else:
         raise ValueError(f"Expected data_sample to have 2 or 3 dimensions, but got {data_sample.dim()}")


    num_features = data_sample.shape[0]
    sequence_length = data_sample.shape[1]


    # Denormalize if dataset provided and has denormalize method
    if dataset is not None and hasattr(dataset, 'denormalize'):
        try:
            # Denormalize expects [features, time] or [batch=1, features, time]
            # Pass a view with batch dimension for consistent denormalization method
            data_sample_denorm = dataset.denormalize(data_sample.unsqueeze(0)).squeeze(0)
        except Exception as e:
            print(f"Warning: Could not denormalize data for plotting: {e}")
            data_sample_denorm = data_sample # Use original if denormalization fails
    else:
        data_sample_denorm = data_sample # Use original if no dataset or no denormalize


    fig, axes = plt.subplots(num_features, 1, figsize=(12, 3*num_features))
    if num_features == 1:
        axes = [axes] # Ensure axes is always a list

    for i in range(num_features):
        axes[i].plot(data_sample_denorm[i])

        # Safely get column name
        col_name = f'Dimension {i+1}'
        if dataset is not None and hasattr(dataset, 'column_names') and len(dataset.column_names) > i:
             col_name = dataset.column_names[i]

        axes[i].set_title(f'{title} - {col_name}')
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()


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