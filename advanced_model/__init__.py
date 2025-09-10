"""
Time Series Diffusion Model Package

A complete implementation of diffusion models for time series data including:
- Data loading and preprocessing for financial/time series data
- WaveNet-based diffusion model architecture
- Training utilities and forward/reverse diffusion processes
- Sampling and generation functions
- Comprehensive evaluation metrics (CRPS, distance metrics)
"""

# Core data handling
from .data_loader import (
    load_yf_data,
    show_yf_data,
    TimeSeriesDataset,
    load_timeseries_dataset,
    show_timeseries_sample,
    SEQUENCE_LENGTH,
    INPUT_DIM,
    BATCH_SIZE
)

# Model architecture
from .model import (
    WaveNetDiffusion,
    ResidualBlock,
    SinusoidalPositionEmbeddings,
    create_wavenet_model
)

# Forward diffusion process
from .forward_process import (
    linear_beta_schedule,
    get_index_from_list,
    forward_diffusion_sample,
    T,  # Default timesteps
    betas,
    alphas,
    alphas_cumprod,
    alphas_cumprod_prev,
    sqrt_recip_alphas,
    sqrt_alphas_cumprod,
    sqrt_one_minus_alphas_cumprod,
    posterior_variance
)

# Sampling and generation
from .data_sampler import (
    sample_timestep,
    sample_plot_timeseries,
    sample_multiple_timeseries,
    sample_single_timeseries
)

# Training utilities
from .trainer import (
    get_loss,
    device
)

# Evaluation framework
from .evaluator import (
    DiffusionModelEvaluator,
    evaluate_trained_model
)

# Global configuration variables
# Auto-calculated based on your data (These will be set when the dataset is created)
SEQUENCE_LENGTH = None  # Will be set automatically
INPUT_DIM = None        # Will be set automatically
BATCH_SIZE = 128

# Package metadata
__version__ = "1.0.0"
__author__ = "Your Name"
__description__ = "Time Series Diffusion Models for Financial Data"

# Define what gets imported with "from base import *"
__all__ = [
    # Data loading
    'load_yf_data',
    'show_yf_data',
    'TimeSeriesDataset',
    'load_timeseries_dataset',
    'show_timeseries_sample',
    'SEQUENCE_LENGTH',
    'INPUT_DIM', 
    'BATCH_SIZE',
    
    # Model
    'WaveNetDiffusion',
    'create_wavenet_model',
    'ResidualBlock',
    'SinusoidalPositionEmbeddings',
    
    # Forward process
    'linear_beta_schedule',
    'get_index_from_list',
    'forward_diffusion_sample',
    'T',
    'betas',
    'alphas',
    'alphas_cumprod',
    'sqrt_alphas_cumprod',
    'sqrt_one_minus_alphas_cumprod',
    'posterior_variance',
    
    # Sampling
    'sample_timestep',
    'sample_plot_timeseries', 
    'sample_multiple_timeseries',
    'sample_single_timeseries',
    
    # Training
    'get_loss',
    'device',
    
    # Evaluation
    'DiffusionModelEvaluator',
    'evaluate_trained_model'
]

# Convenience functions for quick setup
def quick_setup_stock_data(ticker="^GSPC", start_date="2024-01-01", sequence_length=60, columns=None):
    """
    Quick setup function to load and prepare stock data for diffusion modeling.
    
    Args:
        ticker: Stock ticker symbol (default: S&P 500)
        start_date: Start date for data download
        sequence_length: Length of sequences for training
        columns: List of columns to use (default: OHLC)
    
    Returns:
        tuple: (dataset, dataloader, info_dict)
    """
    if columns is None:
        columns = ['Open', 'High', 'Low', 'Close']
    
    # Load data
    yf_df = load_yf_data(ticker, start_date)
    
    # Create dataset
    dataset, dataloader, info = load_timeseries_dataset(
        yf_dataframe=yf_df,
        use_columns=columns,
        sequence_length=sequence_length
    )
    
    return dataset, dataloader, info

def quick_create_model(input_dim=4, **kwargs):
    """
    Quick model creation with sensible defaults.
    
    Args:
        input_dim: Number of input features
        **kwargs: Additional arguments for WaveNetDiffusion
    
    Returns:
        WaveNetDiffusion model
    """
    return create_wavenet_model(input_dim=input_dim)

# Example usage docstring
def example_usage():
    """
    Example usage of the package:
    
    # Quick setup
    from base import quick_setup_stock_data, quick_create_model
    
    # Load S&P 500 data
    dataset, dataloader, info = quick_setup_stock_data()
    
    # Create model
    model = quick_create_model(input_dim=4)
    
    # Train (see trainer.py for full training loop)
    # ... training code ...
    
    # Generate samples
    from base import sample_single_timeseries
    sample_single_timeseries(model, dataset)
    
    # Evaluate
    from base import evaluate_trained_model
    results = evaluate_trained_model(model, dataset)
    """
    pass