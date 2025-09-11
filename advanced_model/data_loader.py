import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
import numpy as np
import config

# GluonTS imports with proper multivariate handling
try:
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.field_names import FieldName
    from gluonts.dataset.multivariate_grouper import MultivariateGrouper
    from gluonts.transform import (
        Chain, AddTimeFeatures, AddAgeFeature, AddObservedValuesIndicator,
        AsNumpyArray, ExpandDimArray, InstanceSplitter, TestSplitSampler
    )
    from gluonts.time_feature import get_seasonality
    GLUONTS_AVAILABLE = True
    print("GluonTS available with multivariate support")
except ImportError:
    GLUONTS_AVAILABLE = False
    print("Warning: GluonTS not available. Enhanced features disabled.")

from typing import Optional, List, Dict, Any, Tuple
import warnings

# Download yf data
def load_yf_data(ticker_name="^GSPC", start_date="2024-01-01"):
    """ 
    Get data from yfinance with enhanced data quality checks
    """
    data = yf.download(ticker_name, start=start_date)
    yfDf = pd.DataFrame(data)
    yfDf.reset_index(inplace=True)
    
    print(f"Downloaded {len(yfDf)} rows for {ticker_name}")
    print(f"Date range: {yfDf['Date'].min()} to {yfDf['Date'].max()}")
    
    missing_values = yfDf.isnull().sum()
    if missing_values.any():
        print("Warning: Missing values detected:")
        print(missing_values[missing_values > 0])
    
    return yfDf

def show_yf_data(dataset, num_samples=20):
    """Show data from yfinance with enhanced visualization"""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    
    # Price plot
    dataset.plot(x='Date', y='Close', ax=ax1, color='blue', linewidth=1.5)
    ax1.set_title('Stock Price Data - Close Price')
    ax1.set_ylabel('Price')
    ax1.grid(True, alpha=0.3)
    
    # Volume plot
    if 'Volume' in dataset.columns:
        dataset.plot(x='Date', y='Volume', ax=ax2, color='orange', alpha=0.7)
        ax2.set_title('Trading Volume')
        ax2.set_ylabel('Volume')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

class TimeSeriesDataset(Dataset):
    """
    Enhanced dataset for multivariate time series with proper GluonTS integration
    """
    def __init__(self, 
                 data_path=None, 
                 data=None, 
                 yf_dataframe=None, 
                 use_columns=None, 
                 sequence_length=None, 
                 normalize=True,
                 use_gluonts=True,
                 freq="D"):
        """
        Initialize dataset with fixed GluonTS multivariate support
        """
        self.use_gluonts = use_gluonts and GLUONTS_AVAILABLE
        self.freq = freq
        self.sequence_length = sequence_length
        self.normalize = normalize
        
        # Load and prepare raw data
        self._load_raw_data(data_path, data, yf_dataframe, use_columns)
        
        if self.use_gluonts and yf_dataframe is not None:
            # Use GluonTS for professional preprocessing
            self._setup_gluonts_processing_fixed()
        else:
            # Fallback to original processing
            self._setup_traditional_processing()
            
    def _load_raw_data(self, data_path, data, yf_dataframe, use_columns):
        """Load raw data from various sources"""
        if yf_dataframe is not None:
            if use_columns is None:
                # Default to OHLCV for multivariate analysis
                available_cols = yf_dataframe.columns.tolist()
                preferred_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
                use_columns = [col for col in preferred_cols if col in available_cols]
                
                if not use_columns:
                    exclude_cols = ['Date', 'date', 'DateTime', 'datetime', 'Datetime', 'timestamp', 'Timestamp']
                    potential_cols = [col for col in available_cols if col not in exclude_cols]
                    numeric_data = yf_dataframe[potential_cols].select_dtypes(include=[np.number])
                    use_columns = numeric_data.columns.tolist()

            print(f"Using columns for multivariate analysis: {use_columns}")
            self.raw_data = yf_dataframe[use_columns].values.astype(np.float32)
            self.column_names = use_columns
            self.dates = pd.to_datetime(yf_dataframe['Date']) if 'Date' in yf_dataframe.columns else None
            
        elif data_path is not None:
            df = pd.read_csv(data_path)
            self.raw_data = df.values.astype(np.float32)
            self.column_names = df.columns.tolist()
            self.dates = None
            
        elif data is not None:
            self.raw_data = np.array(data, dtype=np.float32)
            if len(self.raw_data.shape) > 1:
                self.column_names = [f"Feature_{i}" for i in range(self.raw_data.shape[1])]
            else:
                self.column_names = ["Feature_0"]
            self.dates = None
        else:
            self.raw_data = self._generate_synthetic_data()
            self.column_names = ["Open", "High", "Low", "Close", "Volume"]
            self.dates = None
    
    def _setup_gluonts_processing_fixed(self):
        """
        FIXED: Setup GluonTS preprocessing for multivariate data
        
        The key fix: Create separate univariate series for each feature, 
        then use MultivariateGrouper to combine them properly.
        """
        print("Setting up FIXED GluonTS preprocessing for multivariate time series...")
        
        if self.dates is None:
            print("Creating synthetic date index")
            self.dates = pd.date_range(start='2020-01-01', periods=len(self.raw_data), freq=self.freq)
        
        # FIXED: Create separate univariate series for each feature
        univariate_series = []
        
        for i, feature_name in enumerate(self.column_names):
            # Each feature becomes a separate univariate series
            univariate_data = {
                FieldName.TARGET: self.raw_data[:, i],  # 1D array for this feature
                FieldName.START: self.dates[0],
                FieldName.ITEM_ID: f"{feature_name}_series",  # Unique ID for each feature
                "feat_static_cat": [i]  # Optional: category for this feature
            }
            univariate_series.append(univariate_data)
        
        # Create ListDataset with univariate series
        univariate_dataset = ListDataset(univariate_series, freq=self.freq)
        
        # FIXED: Use MultivariateGrouper to properly combine series
        if len(self.column_names) > 1:
            try:
                grouper = MultivariateGrouper(max_target_dim=len(self.column_names))
                self.gluonts_dataset = grouper(univariate_dataset)
                
                # Extract the grouped multivariate data
                multivariate_data = list(self.gluonts_dataset)[0]
                self.data = multivariate_data[FieldName.TARGET].T  # [time, features]
                
                print(f"GluonTS multivariate grouping successful:")
                print(f"  - Original shape: {self.raw_data.shape}")
                print(f"  - Processed shape: {self.data.shape}")
                
            except Exception as e:
                print(f"MultivariateGrouper failed: {e}")
                print("Falling back to simple processing...")
                self.data = self.raw_data
                
        else:
            # Single feature case
            self.data = self.raw_data
        
        # Setup sequence parameters
        self._setup_sequences()
        
        # Apply normalization if requested
        if self.normalize:
            self._apply_normalization()
    
    def _setup_traditional_processing(self):
        """Fallback to traditional processing without GluonTS"""
        print("Using traditional preprocessing (GluonTS disabled or unavailable)")
        
        self.data = self.raw_data
        
        # Setup sequences
        self._setup_sequences()
        
        # Apply normalization
        if self.normalize:
            self._apply_normalization()
    
    def _setup_sequences(self):
        """Setup sequence windowing parameters"""
        # Calculate dimensions
        self.total_length = self.data.shape[0]
        self.input_dim = self.data.shape[1] if len(self.data.shape) > 1 else 1
        
        # Ensure data is 2D
        if len(self.data.shape) == 1:
            self.data = self.data.reshape(-1, 1)
            
        # Set sequence length
        if self.sequence_length is None:
            self.sequence_length = self.total_length
            self.num_sequences = 1
        else:
            self.sequence_length = min(self.sequence_length, self.total_length)
            self.num_sequences = max(1, self.total_length - self.sequence_length + 1)
        
        print(f"Sequence setup: {self.num_sequences} sequences of length {self.sequence_length}")
    
    def _apply_normalization(self):
        """Apply normalization to [-1, 1] range"""
        self.data_min = self.data.min(axis=0, keepdims=True)  # [1, features]
        self.data_max = self.data.max(axis=0, keepdims=True)  # [1, features]
        
        # Reshape for broadcasting
        self.data_min = self.data_min.squeeze(0)  # [features]
        self.data_max = self.data_max.squeeze(0)  # [features]
        
        # Apply min-max normalization to [-1, 1]
        data_range = self.data_max - self.data_min
        # Avoid division by zero
        data_range = np.where(data_range == 0, 1, data_range)
        
        self.data = 2 * (self.data - self.data_min) / data_range - 1
        
        print(f"Normalization applied:")
        print(f"  - Min values: {self.data_min}")
        print(f"  - Max values: {self.data_max}")
        print(f"  - Normalized data range: [{self.data.min():.3f}, {self.data.max():.3f}]")
    
    def _generate_synthetic_data(self, length=1000, dim=5):
        """Generate synthetic multivariate financial data"""
        t = np.linspace(0, 10, length)
        
        # Base price trend
        base_price = 100 + 20 * np.sin(2 * np.pi * 0.1 * t) + 0.5 * t
        noise = np.random.randn(length) * 2
        
        # Generate OHLC with realistic relationships
        close_prices = base_price + noise
        
        # Open prices (close from previous day + gap)
        open_prices = np.roll(close_prices, 1)
        open_prices[0] = close_prices[0]
        open_prices += np.random.randn(length) * 0.5
        
        # High and Low prices
        daily_range = np.abs(np.random.randn(length)) * 3 + 1
        high_prices = np.maximum(open_prices, close_prices) + daily_range * 0.6
        low_prices = np.minimum(open_prices, close_prices) - daily_range * 0.4
        
        # Volume
        volatility = np.abs(high_prices - low_prices)
        volume = (1000000 + 500000 * volatility / volatility.mean() + 
                 np.random.randn(length) * 200000)
        volume = np.maximum(volume, 100000)
        
        # Combine into OHLCV
        data = np.column_stack([
            open_prices, high_prices, low_prices, close_prices, volume
        ])
        
        return data.astype(np.float32)

    def __len__(self):
        return self.num_sequences

    def __getitem__(self, idx):
        """Get a single sequence for training"""
        if self.num_sequences == 1:
            sequence = self.data
        else:
            start_idx = idx
            end_idx = start_idx + self.sequence_length
            sequence = self.data[start_idx:end_idx]

        # Convert to tensor and transpose for Conv1d: [time, features] → [features, time]
        tensor = torch.FloatTensor(sequence)
        if len(tensor.shape) == 2 and tensor.shape[1] == self.input_dim:
            tensor = tensor.transpose(0, 1)

        return tensor

    def denormalize(self, normalized_data):
        """Convert normalized data back to original scale"""
        if not hasattr(self, 'data_min') or self.data_min is None:
            return normalized_data
            
        if isinstance(normalized_data, torch.Tensor):
            normalized_data = normalized_data.detach().cpu().numpy()

        # Reshape for proper broadcasting: [features, 1]
        data_min = self.data_min[:, np.newaxis]
        data_max = self.data_max[:, np.newaxis]
        data_range = data_max - data_min
        data_range = np.where(data_range == 0, 1, data_range)

        # Reverse normalization
        return (normalized_data + 1) / 2 * data_range + data_min

def load_timeseries_dataset(data_path=None, 
                          data=None, 
                          yf_dataframe=None, 
                          use_columns=None, 
                          sequence_length=None,
                          use_gluonts=True,
                          freq="D"):
    """
    FIXED: Load and prepare multivariate time series dataset with proper GluonTS handling
    """
    print("Loading multivariate time series dataset with FIXED GluonTS integration...")
    
    # Create enhanced dataset with fixed GluonTS integration
    dataset = TimeSeriesDataset(
        data_path=data_path, 
        data=data, 
        yf_dataframe=yf_dataframe, 
        use_columns=use_columns, 
        sequence_length=sequence_length,
        use_gluonts=use_gluonts,
        freq=freq
    )

    # Update global configuration
    config.SEQUENCE_LENGTH = dataset.sequence_length
    config.INPUT_DIM = dataset.input_dim

    # Create PyTorch DataLoader
    dataloader = DataLoader(
        dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        drop_last=True
    )

    # Create comprehensive dataset information
    data_range = '[-1, 1] (normalized)' if dataset.normalize else 'original'
    data_info = {
        'sequence_length': config.SEQUENCE_LENGTH,
        'input_dim': config.INPUT_DIM,
        'total_sequences': len(dataset),
        'batch_size': config.BATCH_SIZE,
        'data_range': data_range,
        'columns': dataset.column_names,
        'use_gluonts': use_gluonts and GLUONTS_AVAILABLE,
        'frequency': freq,
        'has_time_features': hasattr(dataset, 'gluonts_dataset'),
        'missing_data_handling': 'GluonTS multivariate grouper' if use_gluonts and GLUONTS_AVAILABLE else 'None'
    }

    print(f"Dataset loaded successfully:")
    print(f"  - Multivariate features: {dataset.column_names}")
    print(f"  - Input dimension: {config.INPUT_DIM}")
    print(f"  - Sequence length: {config.SEQUENCE_LENGTH}")
    print(f"  - Total sequences: {len(dataset)}")
    print(f"  - GluonTS preprocessing: {data_info['use_gluonts']}")

    return dataset, dataloader, data_info

def show_timeseries_sample(data_sample, dataset=None, title="Multivariate Time Series Sample"):
    """Visualize multivariate time series data with enhanced plotting for OHLCV"""
    # Handle tensor format
    if torch.is_tensor(data_sample):
        data_sample = data_sample.detach().cpu()

    # Handle batch dimension
    if data_sample.dim() == 3:
        data_sample = data_sample[0]  # Take first sample from batch
        
    # Ensure correct orientation: [features, time]
    if data_sample.shape[0] != config.INPUT_DIM and data_sample.shape[1] == config.INPUT_DIM:
        data_sample = data_sample.transpose(0, 1)
    elif data_sample.dim() != 2:
        raise ValueError(f"Expected 2D or 3D tensor, got {data_sample.dim()}D")

    num_features = data_sample.shape[0]

    # Denormalize if possible
    if dataset is not None and hasattr(dataset, 'denormalize'):
        try:
            data_sample_denorm = dataset.denormalize(data_sample.unsqueeze(0)).squeeze(0)
        except Exception as e:
            print(f"Warning: Could not denormalize data for plotting: {e}")
            data_sample_denorm = data_sample
    else:
        data_sample_denorm = data_sample

    # Enhanced plotting for financial data
    if dataset and hasattr(dataset, 'column_names'):
        column_names = dataset.column_names
        
        # Check if this looks like OHLCV data
        is_ohlcv = any(col.lower() in ['open', 'high', 'low', 'close'] for col in column_names)
        
        if is_ohlcv and num_features >= 4:
            _plot_ohlcv_style(data_sample_denorm, column_names, title)
        else:
            _plot_multivariate_standard(data_sample_denorm, column_names, title)
    else:
        _plot_multivariate_standard(data_sample_denorm, 
                                   [f'Feature {i+1}' for i in range(num_features)], 
                                   title)

def _plot_ohlcv_style(data_sample, column_names, title):
    """Enhanced plotting for OHLCV data"""
    # Find OHLCV indices
    ohlc_indices = {}
    volume_idx = None
    
    for i, col in enumerate(column_names):
        col_lower = col.lower()
        if 'open' in col_lower:
            ohlc_indices['open'] = i
        elif 'high' in col_lower:
            ohlc_indices['high'] = i
        elif 'low' in col_lower:
            ohlc_indices['low'] = i
        elif 'close' in col_lower:
            ohlc_indices['close'] = i
        elif 'volume' in col_lower:
            volume_idx = i

    # Create figure with subplots
    fig_height = 10 if volume_idx is not None else 8
    fig, axes = plt.subplots(2 if volume_idx is not None else 1, 1, 
                            figsize=(15, fig_height), sharex=True)
    
    if volume_idx is not None:
        price_ax, volume_ax = axes
    else:
        price_ax = axes
        volume_ax = None

    # Plot OHLC data
    x = range(data_sample.shape[1])
    
    if len(ohlc_indices) >= 4:
        if 'open' in ohlc_indices:
            price_ax.plot(x, data_sample[ohlc_indices['open']], label='Open', alpha=0.8)
        if 'high' in ohlc_indices:
            price_ax.plot(x, data_sample[ohlc_indices['high']], label='High', alpha=0.8)
        if 'low' in ohlc_indices:
            price_ax.plot(x, data_sample[ohlc_indices['low']], label='Low', alpha=0.8)
        if 'close' in ohlc_indices:
            price_ax.plot(x, data_sample[ohlc_indices['close']], label='Close', linewidth=2)
    
    price_ax.set_title(f'{title} - OHLC Prices')
    price_ax.legend()
    price_ax.grid(True, alpha=0.3)
    
    # Plot volume if available
    if volume_idx is not None and volume_ax is not None:
        volume_ax.plot(x, data_sample[volume_idx], color='orange', alpha=0.7)
        volume_ax.set_title(f'{title} - Volume')
        volume_ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()

def _plot_multivariate_standard(data_sample, column_names, title):
    """Standard multivariate plotting"""
    num_features = data_sample.shape[0]
    
    fig, axes = plt.subplots(num_features, 1, figsize=(15, 3*num_features), sharex=True)
    if num_features == 1:
        axes = [axes]

    for i in range(num_features):
        axes[i].plot(data_sample[i], linewidth=1.5)
        axes[i].set_title(f'{title} - {column_names[i]}')
        axes[i].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.show()

# Example usage with enhanced documentation
"""
FIXED example usage for multivariate OHLCV data:

# Load financial data with FIXED GluonTS preprocessing
yf_df = load_yf_data("^GSPC", "2023-01-01")
show_yf_data(yf_df)

# Create multivariate dataset with FIXED GluonTS integration
dataset, dataloader, info = load_timeseries_dataset(
    yf_dataframe=yf_df,
    use_columns=['Open', 'High', 'Low', 'Close', 'Volume'],  # Multivariate inputs
    sequence_length=60,
    use_gluonts=True,  # Enable FIXED preprocessing
    freq="D"  # Daily frequency
)

# This will now work without the shape error!
"""