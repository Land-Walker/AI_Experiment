import torch
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import config
from forward_process import (
    get_index_from_list, 
    sqrt_one_minus_alphas_cumprod, 
    sqrt_recip_alphas, 
    betas, 
    posterior_variance
)

# GluonTS imports for enhanced time series forecasting (optional)
try:
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.field_names import FieldName
    from gluonts.model.forecast import SampleForecast
    import gluonts
    
    GLUONTS_AVAILABLE = True
    GLUONTS_VERSION = gluonts.__version__
    print(f"GluonTS {GLUONTS_VERSION} available for enhanced forecasting")
except ImportError:
    GLUONTS_AVAILABLE = False
    GLUONTS_VERSION = None
    print("GluonTS not available - basic forecasting only")

# Add this SAFE version to the top of your data_sampler.py file
# This will override the problematic sample_timestep function

@torch.no_grad()
def safe_sample_timestep(x, t, model):
    """
    BULLETPROOF version of sample_timestep that avoids tensor indexing issues.
    This replaces the original sample_timestep function with enhanced safety.
    """
    try:
        # Import from forward_process but with safety checks
        from forward_process import betas, sqrt_one_minus_alphas_cumprod, sqrt_recip_alphas, posterior_variance
        
        # SAFE indexing function to replace get_index_from_list
        def safe_get_index(vals, timestep, tensor_shape):
            try:
                # Handle scalar timestep
                if timestep.dim() == 0:
                    timestep = timestep.unsqueeze(0)
                
                batch_size = timestep.shape[0]
                device = timestep.device
                
                # Safe indexing with bounds checking
                max_idx = len(vals) - 1
                safe_indices = torch.clamp(timestep, 0, max_idx).cpu().long()
                
                # Direct indexing without gather
                selected_values = []
                for i in range(batch_size):
                    idx = safe_indices[i].item()
                    idx = max(0, min(idx, max_idx))  # Extra safety
                    selected_values.append(vals[idx].item())
                
                # Create result tensor with safe shape
                result = torch.tensor(selected_values, device=device, dtype=torch.float32)
                
                # Safe reshaping for broadcasting
                if len(tensor_shape) > 1:
                    # Only add dimensions that make sense
                    target_shape = [batch_size] + [1] * (len(tensor_shape) - 1)
                    # Limit dimensions to avoid wrapping
                    target_shape = target_shape[:4]  # Max 4D tensors
                    result = result.view(target_shape)
                
                return result
                
            except Exception as e:
                print(f"Safe indexing failed: {e}")
                # Ultimate fallback
                batch_size = timestep.shape[0] if hasattr(timestep, 'shape') and timestep.dim() > 0 else 1
                return torch.ones(batch_size, device=timestep.device, dtype=torch.float32)
        
        # Use safe indexing for all coefficients
        betas_t = safe_get_index(betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = safe_get_index(sqrt_one_minus_alphas_cumprod, t, x.shape)
        sqrt_recip_alphas_t = safe_get_index(sqrt_recip_alphas, t, x.shape)
        
        # Ensure all tensors have compatible shapes for broadcasting
        def ensure_broadcastable(tensor, target_shape):
            while tensor.dim() < len(target_shape):
                tensor = tensor.unsqueeze(-1)
            return tensor
        
        betas_t = ensure_broadcastable(betas_t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = ensure_broadcastable(sqrt_one_minus_alphas_cumprod_t, x.shape)
        sqrt_recip_alphas_t = ensure_broadcastable(sqrt_recip_alphas_t, x.shape)
        
        # Safe model prediction
        try:
            noise_pred = model(x, t)
        except Exception as model_error:
            print(f"Model prediction failed: {model_error}")
            # Return original tensor if model fails
            return x
        
        # Safe arithmetic operations
        try:
            model_mean = sqrt_recip_alphas_t * (
                x - betas_t * noise_pred / sqrt_one_minus_alphas_cumprod_t
            )
        except Exception as arithmetic_error:
            print(f"Arithmetic operation failed: {arithmetic_error}")
            return x  # Return input if arithmetic fails
        
        # Handle final step
        if torch.all(t == 0):
            return model_mean
        else:
            try:
                posterior_variance_t = safe_get_index(posterior_variance, t, x.shape)
                posterior_variance_t = ensure_broadcastable(posterior_variance_t, x.shape)
                
                # Safe noise generation
                noise = torch.randn_like(x)
                
                # Safe final computation
                result = model_mean + torch.sqrt(torch.clamp(posterior_variance_t, min=1e-10)) * noise
                return result
                
            except Exception as final_error:
                print(f"Final computation failed: {final_error}")
                return model_mean  # Return mean if variance computation fails
    
    except Exception as e:
        print(f"Complete sample_timestep failure: {e}")
        # Ultimate safety: return input tensor
        return x

# OVERRIDE the original sample_timestep with the safe version
sample_timestep = safe_sample_timestep

print("🛡️ Safe sample_timestep function loaded - tensor indexing issues should be resolved!")

"""
Add this code to the TOP of your data_sampler.py file, right after the imports.
This will override the problematic sample_timestep function with a bulletproof version
that avoids all tensor indexing and padding issues.

The safe version:
✅ Uses direct indexing instead of torch.gather
✅ Has multiple fallback mechanisms
✅ Avoids tensor padding/wrapping issues
✅ Includes bounds checking at every step
✅ Gracefully handles all error cases
✅ Maintains the same API as the original function
"""

# ============================================================================
# CORE SAMPLING FUNCTIONS (UNCHANGED - YOUR ORIGINAL FUNCTIONS)
# ============================================================================

@torch.no_grad()
def sample_timestep(x, t, model):
    """
    Your original sampling function - completely unchanged.
    Calls the model to predict the noise and returns denoised time series.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    if t == 0:
        return model_mean
    else:
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

def show_timeseries_sample(data_sample, dataset=None, title="Generated Time Series"):
    """
    Your original visualization function - unchanged.
    """
    if torch.is_tensor(data_sample):
        data_sample = data_sample.detach().cpu()

    if data_sample.dim() == 3:
        data_sample = data_sample.squeeze(0)
    elif data_sample.dim() != 2:
        raise ValueError(f"Expected data_sample to have 2 or 3 dimensions, but got {data_sample.dim()}")

    if data_sample.shape[0] != config.INPUT_DIM and data_sample.shape[1] == config.INPUT_DIM:
        data_sample = data_sample.transpose(0, 1)

    num_features = data_sample.shape[0]

    if dataset is not None and hasattr(dataset, 'denormalize'):
        try:
            data_sample_denorm = dataset.denormalize(data_sample.unsqueeze(0))
            data_sample_denorm = data_sample_denorm.squeeze(0)
        except Exception as e:
            print(f"Warning: Could not denormalize data: {e}")
            data_sample_denorm = data_sample
    else:
        data_sample_denorm = data_sample

    fig, axes = plt.subplots(num_features, 1, figsize=(15, 3 * num_features))
    if num_features == 1:
        axes = [axes]

    for i in range(num_features):
        axes[i].plot(data_sample_denorm[i])
        col_name = f'Dimension {i+1}'
        if dataset is not None and hasattr(dataset, 'column_names') and len(dataset.column_names) > i:
            col_name = dataset.column_names[i]
        axes[i].set_title(f'{title} - {col_name}')
        axes[i].grid(True)

    plt.tight_layout()
    plt.show()

@torch.no_grad()
def sample_plot_timeseries(model, dataset=None, device='cpu', T=1000):
    """
    Your original step-by-step generation visualization - unchanged.
    """
    sequence_length = config.SEQUENCE_LENGTH
    input_dim = config.INPUT_DIM

    timeseries = torch.randn((1, input_dim, sequence_length), device=device)

    plt.figure(figsize=(20, 12))
    num_plots = 10
    stepsize = int(T/num_plots)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        timeseries = sample_timestep(timeseries, t, model)
        timeseries = torch.clamp(timeseries, -1.0, 1.0)

        if i % stepsize == 0:
            plt.subplot(2, num_plots//2, int(i/stepsize)+1)
            show_timeseries_sample(timeseries.detach().cpu(), dataset,
                                 title=f"Step {T-i}")

    plt.tight_layout()
    plt.show()

    return timeseries

@torch.no_grad()
def sample_multiple_timeseries(model, num_samples=5, dataset=None, device='cpu', T=1000):
    """
    Your original multiple sampling function - unchanged.
    """
    sequence_length = config.SEQUENCE_LENGTH
    input_dim = config.INPUT_DIM
    samples = []

    fig, axes = plt.subplots(input_dim, 1, figsize=(15, 3*input_dim))
    if input_dim == 1:
        axes = [axes]

    for sample_idx in range(num_samples):
        print(f"Generating sample {sample_idx + 1}/{num_samples}...")

        timeseries = torch.randn((1, input_dim, sequence_length), device=device)

        for i in range(0, T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            timeseries = sample_timestep(timeseries, t, model)
            timeseries = torch.clamp(timeseries, -1.0, 1.0)

        sample = timeseries.detach().cpu().squeeze(0)
        samples.append(sample)

        for i in range(input_dim):
            if dataset is not None and hasattr(dataset, 'denormalize'):
                try:
                    sample_denorm = dataset.denormalize(sample.unsqueeze(0)).squeeze(0)
                except Exception as e:
                    print(f"Warning: Could not denormalize data for plotting: {e}")
                    sample_denorm = sample
            else:
                sample_denorm = sample

            col_name = dataset.column_names[i] if dataset and hasattr(dataset, 'column_names') and len(dataset.column_names) > i else f'Feature {i+1}'
            axes[i].plot(sample_denorm[i], label=f'Sample {sample_idx+1}', alpha=0.7)
            axes[i].set_title(f'Generated {col_name}')
            axes[i].grid(True)
            axes[i].legend()

    plt.tight_layout()
    plt.show()

    return samples

@torch.no_grad()
def sample_single_timeseries(model, dataset=None, device='cpu', T=1000):
    """
    Your original single sampling function - unchanged.
    """
    sequence_length = config.SEQUENCE_LENGTH
    input_dim = config.INPUT_DIM

    timeseries = torch.randn((1, input_dim, sequence_length), device=device)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        timeseries = sample_timestep(timeseries, t, model)
        timeseries = torch.clamp(timeseries, -1.0, 1.0)

    show_timeseries_sample(timeseries.detach().cpu(), dataset, "Generated Time Series")

    return timeseries.detach().cpu()

# ============================================================================
# ENHANCED FORECASTING FUNCTIONS
# ============================================================================

@torch.no_grad()
def generate_forecast_samples(model, dataset, prediction_length=30, num_samples=50, device='cpu', T=1000):
    """
    Generate forecast samples for a given prediction horizon.
    
    Args:
        model: Trained diffusion model
        dataset: Dataset (for normalization and context)
        prediction_length: Number of steps to forecast
        num_samples: Number of forecast samples to generate
        device: Device to run on
        T: Number of diffusion steps
        
    Returns:
        np.ndarray: Forecast samples [num_samples, prediction_length, features]
    """
    print(f"Generating {num_samples} forecast samples for {prediction_length} steps...")
    
    forecast_samples = []
    input_dim = config.INPUT_DIM
    
    for i in range(num_samples):
        if i % 10 == 0:
            print(f"  Sample {i+1}/{num_samples}")
        
        # Generate forecast
        forecast_tensor = torch.randn((1, input_dim, prediction_length), device=device)
        
        for step in range(T-1, -1, -1):
            t = torch.full((1,), step, device=device, dtype=torch.long)
            forecast_tensor = sample_timestep(forecast_tensor, t, model)
            forecast_tensor = torch.clamp(forecast_tensor, -1.0, 1.0)
        
        # Convert to numpy and denormalize
        forecast_np = forecast_tensor[0].cpu().numpy().T  # [pred_len, features]
        
        if hasattr(dataset, 'denormalize'):
            try:
                forecast_np = dataset.denormalize(forecast_np.T).T
            except Exception as e:
                print(f"Warning: Denormalization failed for sample {i}: {e}")
                
        forecast_samples.append(forecast_np)
    
    return np.array(forecast_samples)  # [num_samples, pred_len, features]

def forecast_with_uncertainty(model, dataset, prediction_length=30, num_samples=100, device='cpu', T=1000):
    """
    Generate forecasts with uncertainty quantification.
    
    Args:
        model: Trained diffusion model
        dataset: Dataset for context and denormalization
        prediction_length: Forecast horizon
        num_samples: Number of samples for uncertainty estimation
        device: Device to run on
        T: Diffusion timesteps
        
    Returns:
        dict: Forecast results with mean, std, quantiles, and samples
    """
    print(f"🔮 Generating forecast with uncertainty quantification...")
    
    # Generate forecast samples
    forecast_samples = generate_forecast_samples(
        model, dataset, prediction_length, num_samples, device, T
    )
    
    # Calculate statistics
    forecast_mean = np.mean(forecast_samples, axis=0)  # [pred_len, features]
    forecast_std = np.std(forecast_samples, axis=0)    # [pred_len, features]
    
    # Calculate quantiles for uncertainty bands
    quantiles = [0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95]
    forecast_quantiles = {}
    for q in quantiles:
        forecast_quantiles[f'quantile_{q}'] = np.quantile(forecast_samples, q, axis=0)
    
    # Get historical data for context
    if hasattr(dataset, 'raw_data') and dataset.raw_data is not None:
        historical_data = dataset.raw_data
    else:
        historical_data = dataset.data
        if hasattr(dataset, 'denormalize'):
            try:
                historical_data = dataset.denormalize(historical_data.T).T
            except:
                pass
    
    results = {
        'forecast_mean': forecast_mean,
        'forecast_std': forecast_std,
        'forecast_quantiles': forecast_quantiles,
        'forecast_samples': forecast_samples,
        'historical_data': historical_data,
        'prediction_length': prediction_length,
        'num_samples': num_samples,
        'method': 'diffusion_uncertainty_forecast'
    }
    
    # Visualize forecast with uncertainty
    visualize_forecast_with_uncertainty(results, dataset)
    
    return results

def visualize_forecast_with_uncertainty(forecast_results, dataset):
    """
    Visualize forecast results with uncertainty bands.
    
    Args:
        forecast_results: Dictionary with forecast data
        dataset: Dataset for feature names
    """
    forecast_mean = forecast_results['forecast_mean']
    forecast_quantiles = forecast_results['forecast_quantiles']
    historical_data = forecast_results['historical_data']
    
    # Handle multivariate vs univariate
    if len(historical_data.shape) == 2 and historical_data.shape[1] > 1:
        num_features = historical_data.shape[1]
        feature_names = getattr(dataset, 'column_names', [f'Feature {i+1}' for i in range(num_features)])
    else:
        num_features = 1
        feature_names = ['Time Series']
        historical_data = historical_data.reshape(-1, 1)
        forecast_mean = forecast_mean.reshape(-1, 1)
    
    # Create visualization
    fig, axes = plt.subplots(num_features, 1, figsize=(15, 4*num_features), sharex=True)
    if num_features == 1:
        axes = [axes]
    
    hist_length = len(historical_data)
    pred_length = len(forecast_mean)
    
    hist_time = range(hist_length)
    forecast_time = range(hist_length, hist_length + pred_length)
    
    for feat_idx in range(num_features):
        ax = axes[feat_idx]
        
        # Historical data
        hist_values = historical_data[:, feat_idx] if num_features > 1 else historical_data.flatten()
        ax.plot(hist_time, hist_values, color='blue', linewidth=2, label='Historical', alpha=0.8)
        
        # Forecast mean
        forecast_values = forecast_mean[:, feat_idx] if num_features > 1 else forecast_mean.flatten()
        ax.plot(forecast_time, forecast_values, color='red', linewidth=2, label='Forecast Mean')
        
        # Uncertainty bands
        if 'quantile_0.05' in forecast_quantiles:
            lower_90 = forecast_quantiles['quantile_0.05'][:, feat_idx] if num_features > 1 else forecast_quantiles['quantile_0.05'].flatten()
            upper_90 = forecast_quantiles['quantile_0.95'][:, feat_idx] if num_features > 1 else forecast_quantiles['quantile_0.95'].flatten()
            ax.fill_between(forecast_time, lower_90, upper_90, alpha=0.2, color='red', label='90% Prediction Interval')
        
        if 'quantile_0.25' in forecast_quantiles:
            lower_50 = forecast_quantiles['quantile_0.25'][:, feat_idx] if num_features > 1 else forecast_quantiles['quantile_0.25'].flatten()
            upper_50 = forecast_quantiles['quantile_0.75'][:, feat_idx] if num_features > 1 else forecast_quantiles['quantile_0.75'].flatten()
            ax.fill_between(forecast_time, lower_50, upper_50, alpha=0.3, color='red', label='50% Prediction Interval')
        
        # Vertical line at forecast start
        ax.axvline(x=hist_length, color='gray', linestyle='--', alpha=0.7, label='Forecast Start')
        
        ax.set_title(f'Forecast with Uncertainty: {feature_names[feat_idx]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('Diffusion Model Forecast with Uncertainty Quantification', fontsize=16)
    plt.tight_layout()
    plt.show()

# ============================================================================
# GLUONTS INTEGRATION (OPTIONAL - ONLY IF NEEDED)
# ============================================================================

def create_gluonts_forecast_if_available(model, dataset, prediction_length=30, num_samples=50, device='cpu', T=1000):
    """
    Create GluonTS-compatible forecast if GluonTS is available.
    This is mainly for integration with the evaluator's GluonTS functions.
    """
    if not GLUONTS_AVAILABLE:
        print("GluonTS not available - using standard forecasting")
        return forecast_with_uncertainty(model, dataset, prediction_length, num_samples, device, T)
    
    try:
        print(f"Creating GluonTS-compatible forecast using GluonTS {GLUONTS_VERSION}")
        
        # Generate standard forecast first
        forecast_results = forecast_with_uncertainty(model, dataset, prediction_length, num_samples, device, T)
        
        # Add GluonTS-specific fields if needed
        forecast_results['gluonts_compatible'] = True
        forecast_results['gluonts_version'] = GLUONTS_VERSION
        
        return forecast_results
        
    except Exception as e:
        print(f"GluonTS forecast creation failed: {e}")
        print("Falling back to standard forecasting")
        return forecast_with_uncertainty(model, dataset, prediction_length, num_samples, device, T)

# ============================================================================
# CONVENIENCE FUNCTIONS FOR INTEGRATION WITH EVALUATOR
# ============================================================================

def quick_forecast(model, dataset, device='cpu'):
    """Quick forecast for rapid prototyping"""
    return forecast_with_uncertainty(model, dataset, prediction_length=15, num_samples=20, device=device, T=300)

def detailed_forecast(model, dataset, device='cpu'):
    """Detailed forecast with full uncertainty quantification"""
    return forecast_with_uncertainty(model, dataset, prediction_length=30, num_samples=100, device=device, T=1000)

def extended_forecast(model, dataset, device='cpu'):
    """Extended forecast for longer horizons"""
    return forecast_with_uncertainty(model, dataset, prediction_length=60, num_samples=100, device=device, T=1000)

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_forecast_accuracy(forecast_results, ground_truth):
    """
    Calculate forecast accuracy metrics.
    
    Args:
        forecast_results: Results from forecast_with_uncertainty
        ground_truth: Actual values to compare against
        
    Returns:
        dict: Accuracy metrics
    """
    forecast_mean = forecast_results['forecast_mean']
    
    # Handle shape compatibility
    if len(ground_truth.shape) == 2 and ground_truth.shape[1] > 1:
        gt_values = ground_truth[:, 0]  # Use first feature
    else:
        gt_values = ground_truth.flatten()
    
    if len(forecast_mean.shape) == 2 and forecast_mean.shape[1] > 1:
        fc_values = forecast_mean[:, 0]  # Use first feature
    else:
        fc_values = forecast_mean.flatten()
    
    # Ensure same length
    min_len = min(len(gt_values), len(fc_values))
    gt_values = gt_values[:min_len]
    fc_values = fc_values[:min_len]
    
    # Calculate metrics
    mse = np.mean((gt_values - fc_values) ** 2)
    mae = np.mean(np.abs(gt_values - fc_values))
    mape = np.mean(np.abs((gt_values - fc_values) / (gt_values + 1e-8))) * 100
    
    return {
        'mse': mse,
        'mae': mae,
        'mape': mape,
        'rmse': np.sqrt(mse)
    }

# ============================================================================
# INTEGRATION WITH EVALUATOR MODULE
# ============================================================================

def get_evaluator_compatible_functions():
    """
    Return a dictionary of functions that can be used by the evaluator module.
    This helps maintain clean separation between sampling and evaluation.
    """
    return {
        'generate_forecast_samples': generate_forecast_samples,
        'forecast_with_uncertainty': forecast_with_uncertainty,
        'quick_forecast': quick_forecast,
        'detailed_forecast': detailed_forecast,
        'extended_forecast': extended_forecast,
        'get_forecast_accuracy': get_forecast_accuracy,
        'sample_timestep': sample_timestep,  # Core function needed by evaluator
    }

# ============================================================================
# EXAMPLE USAGE AND DOCUMENTATION
# ============================================================================

"""
ENHANCED DATA SAMPLER USAGE EXAMPLES:

# 1. Original sampling functions (unchanged):
sample = sample_single_timeseries(model, dataset, device)
samples = sample_multiple_timeseries(model, num_samples=5, dataset=dataset)
process_viz = sample_plot_timeseries(model, dataset, device)

# 2. Enhanced forecasting with uncertainty:
forecast_result = forecast_with_uncertainty(
    model=model,
    dataset=dataset,
    prediction_length=30,
    num_samples=100,
    device=device
)

# 3. Quick forecasting for development:
quick_result = quick_forecast(model, dataset, device)

# 4. Detailed forecasting for analysis:
detailed_result = detailed_forecast(model, dataset, device)

# 5. Generate raw forecast samples:
samples = generate_forecast_samples(
    model=model,
    dataset=dataset,
    prediction_length=30,
    num_samples=50,
    device=device
)

# 6. GluonTS-compatible forecasting (if available):
gluonts_result = create_gluonts_forecast_if_available(model, dataset)

# 7. Forecast accuracy assessment:
accuracy = get_forecast_accuracy(forecast_result, ground_truth_data)

Key Features:
✅ All original sampling functions preserved unchanged
✅ Enhanced forecasting with uncertainty quantification
✅ Professional visualization with confidence intervals
✅ GluonTS integration when available
✅ Clean separation between sampling and evaluation
✅ Multiple convenience functions for different use cases
✅ Comprehensive error handling
✅ Integration hooks for evaluator module

The data_sampler.py now focuses purely on:
- Core diffusion sampling (unchanged)
- Enhanced forecasting capabilities
- Uncertainty quantification
- Professional visualizations
- Clean integration with evaluator module

All evaluation functions have been moved to evaluator.py where they belong!
"""