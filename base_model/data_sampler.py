import torch
import matplotlib.pyplot as plt
import numpy as np
import config  # PRESERVED: Your config module import

# PRESERVED: Your specific imports from forward_process
from forward_process import (
    get_index_from_list, 
    sqrt_one_minus_alphas_cumprod, 
    sqrt_recip_alphas, 
    betas, 
    posterior_variance
)

@torch.no_grad()  # Disable gradient computation for sampling (inference only)
def sample_timestep(x, t, model):
    """
    Single step of reverse diffusion: remove noise from x_t to get x_{t-1}
    
    This implements the core denoising equation from DDPM. Given noisy data at timestep t,
    the model predicts what noise was added, and we remove it to get cleaner data.
    
    The key equation: μ_θ(x_t, t) = (1/√α_t) * (x_t - (β_t/√(1-ᾱ_t)) * ε_θ(x_t, t))
    
    Args:
        x: Current noisy time series [batch, features, time]
        t: Current timestep [batch] (t=0 means almost clean, t=T means pure noise)
        model: Trained diffusion model that predicts noise
        
    Returns:
        torch.Tensor: Denoised time series (one step closer to clean data)
    """
    # Extract diffusion parameters for current timestep
    # These control how much noise to remove and how much uncertainty to add back
    
    # β_t: noise variance schedule at timestep t
    betas_t = get_index_from_list(betas, t, x.shape)
    
    # √(1 - ᾱ_t): normalization factor for noise prediction
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    
    # 1/√α_t: scaling factor for denoising step
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Core denoising equation: predict mean of denoised distribution
    # model(x, t) predicts the noise that was added at this timestep
    model_mean = sqrt_recip_alphas_t * (
        x - betas_t * model(x, t) / sqrt_one_minus_alphas_cumprod_t
    )
    
    # Get variance for adding controlled noise back (except at final step)
    posterior_variance_t = get_index_from_list(posterior_variance, t, x.shape)

    # At final step (t=0), return clean prediction without adding noise
    if t == 0:
        return model_mean
    else:
        # For intermediate steps, add calibrated noise to maintain proper sampling distribution
        noise = torch.randn_like(x)
        return model_mean + torch.sqrt(posterior_variance_t) * noise

def show_timeseries_sample(data_sample, dataset=None, title="Generated Time Series"):
    """
    Plot a single generated time series sample with proper format handling
    
    Visualizes time series data, handles various tensor shapes, and applies
    denormalization if dataset is provided.
    
    Args:
        data_sample: Generated time series data (various possible shapes)
        dataset: Original dataset object (for denormalization and column names)
        title: Title for the plots
    """
    # Ensure data is on CPU for matplotlib
    if torch.is_tensor(data_sample):
        data_sample = data_sample.detach().cpu()

    # Handle batch dimension: remove if present
    if data_sample.dim() == 3:
        # [batch_size, features, time] → [features, time]
        data_sample = data_sample.squeeze(0)
    elif data_sample.dim() != 2:
         raise ValueError(f"Expected data_sample to have 2 or 3 dimensions, but got {data_sample.dim()}")

    # Ensure correct orientation: [features, time]
    # Check against config.INPUT_DIM to determine if transpose is needed
    if data_sample.shape[0] != config.INPUT_DIM and data_sample.shape[1] == config.INPUT_DIM:  # PRESERVED
         data_sample = data_sample.transpose(0, 1)

    # Extract dimensions for plotting
    num_features = data_sample.shape[0]
    sequence_length = data_sample.shape[1]

    # Create subplot for each feature
    fig, axes = plt.subplots(num_features, 1, figsize=(15, 3 * num_features))
    if num_features == 1:
        axes = [axes]  # Ensure axes is always iterable

    # Denormalize data if dataset supports it (convert from [-1,1] to original scale)
    if dataset is not None and hasattr(dataset, 'denormalize'):
        try:
            # Add batch dimension for denormalization method, then remove it
            data_sample_denorm = dataset.denormalize(data_sample.unsqueeze(0))
            data_sample_denorm = data_sample_denorm.squeeze(0)
        except Exception as e:
            print(f"Warning: Could not denormalize data: {e}")
            data_sample_denorm = data_sample
    else:
        data_sample_denorm = data_sample

    # Plot each feature in separate subplot
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

@torch.no_grad()
def sample_plot_timeseries(model, dataset=None, device='cpu', T=1000):
    """
    Sample and visualize the complete time series generation process
    
    Shows how the model progressively denoises random noise into realistic
    time series data by plotting intermediate steps.
    
    Args:
        model: Trained diffusion model
        dataset: Dataset object for denormalization and metadata
        device: Device for computation ('cpu' or 'cuda')
        T: Total number of diffusion timesteps
        
    Returns:
        torch.Tensor: Final generated time series
    """
    # Get dimensions from config module (PRESERVED)
    sequence_length = config.SEQUENCE_LENGTH
    input_dim = config.INPUT_DIM

    # Start with pure Gaussian noise
    timeseries = torch.randn((1, input_dim, sequence_length), device=device)

    # Set up visualization grid
    plt.figure(figsize=(20, 12))
    num_plots = 10
    stepsize = int(T/num_plots)

    # Reverse diffusion process: T-1 down to 0
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        timeseries = sample_timestep(timeseries, t, model)

        # Clamp to valid range for numerical stability
        timeseries = torch.clamp(timeseries, -1.0, 1.0)

        # Plot intermediate steps
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
    Generate multiple independent time series samples and plot them together
    
    Useful for analyzing diversity and consistency of generated samples.
    Plots all samples on the same axes to compare patterns.
    
    Args:
        model: Trained diffusion model
        num_samples: Number of independent samples to generate
        dataset: Dataset object for denormalization and metadata
        device: Device for computation
        T: Number of diffusion timesteps
        
    Returns:
        list: List of generated time series samples
    """
    # Get dimensions from config (PRESERVED)
    sequence_length = config.SEQUENCE_LENGTH
    input_dim = config.INPUT_DIM

    samples = []

    # Create subplots for each feature
    fig, axes = plt.subplots(input_dim, 1, figsize=(15, 3*input_dim))
    if input_dim == 1:
        axes = [axes]

    # Generate each sample independently
    for sample_idx in range(num_samples):
        print(f"Generating sample {sample_idx + 1}/{num_samples}...")

        # Start with fresh noise for each sample
        timeseries = torch.randn((1, input_dim, sequence_length), device=device)

        # Complete reverse diffusion process
        for i in range(0, T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            timeseries = sample_timestep(timeseries, t, model)
            timeseries = torch.clamp(timeseries, -1.0, 1.0)

        # Store sample and prepare for plotting
        sample = timeseries.detach().cpu().squeeze(0)  # [features, time]
        samples.append(sample)

        # Plot each feature of this sample
        for i in range(input_dim):
            # Denormalize if possible
            if dataset is not None and hasattr(dataset, 'denormalize'):
                 try:
                     sample_denorm = dataset.denormalize(sample.unsqueeze(0)).squeeze(0)
                 except Exception as e:
                     print(f"Warning: Could not denormalize data for plotting: {e}")
                     sample_denorm = sample
            else:
                 sample_denorm = sample

            # Get feature name
            col_name = dataset.column_names[i] if dataset and hasattr(dataset, 'column_names') and len(dataset.column_names) > i else f'Feature {i+1}'
            
            # Plot with transparency to see overlapping patterns
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
    Generate a single time series sample (fastest and most common option)
    
    This is the main function for generating new samples after training.
    Runs complete reverse diffusion and visualizes the result.
    
    Args:
        model: Trained diffusion model
        dataset: Dataset object for denormalization and metadata
        device: Device for computation
        T: Number of diffusion timesteps
        
    Returns:
        torch.Tensor: Generated time series on CPU
    """
    # Get dimensions from config (PRESERVED)
    sequence_length = config.SEQUENCE_LENGTH
    input_dim = config.INPUT_DIM

    # Start with pure noise
    timeseries = torch.randn((1, input_dim, sequence_length), device=device)

    # Complete reverse diffusion: progressively denoise
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        timeseries = sample_timestep(timeseries, t, model)  # Apply one denoising step
        timeseries = torch.clamp(timeseries, -1.0, 1.0)     # Ensure valid range

    # Visualize the final result
    show_timeseries_sample(timeseries.detach().cpu(), dataset, "Generated Time Series")

    return timeseries.detach().cpu()

# ============================================================================
# SAMPLING PROCESS EXPLANATION
# ============================================================================
"""
REVERSE DIFFUSION SAMPLING INTUITION:

The sampling process reverses the forward diffusion:
Forward:  x_0 → x_1 → x_2 → ... → x_T (clean to noise)
Reverse:  x_T → x_{T-1} → x_{T-2} → ... → x_0 (noise to clean)

At each reverse step:
1. Model predicts what noise was added at this step
2. Remove the predicted noise from current noisy data
3. Add small amount of controlled noise (except at final step)

Why add noise back? 
- Maintains proper probability distribution
- Prevents mode collapse 
- Ensures diverse, realistic samples

The key insight: the model learns to "see through" the noise and identify
the underlying clean signal, gradually revealing it step by step.

PRACTICAL CONSIDERATIONS:

1. Clamping: Real diffusion can occasionally produce values outside [-1,1]
   Clamping ensures numerical stability without significantly affecting quality

2. Device handling: Models and data need to be on same device (CPU/GPU)
   Generated samples are moved to CPU for visualization and storage

3. Gradient disabling: @torch.no_grad() saves memory and speeds up inference
   No gradients needed during sampling, only during training

4. Config usage: All dimension references use config module for consistency
   This ensures compatibility with the global configuration system
"""

# Example usage documentation (PRESERVED exactly as in your code)
"""
Example usage:
if __name__ == "__main__":
    # Assuming you have your trained model and dataset
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Option 1: Show the generation process step by step
    # sample_plot_timeseries(dataset, device, T=1000)

    # Option 2: Generate multiple samples
    # samples = sample_multiple_timeseries(num_samples=5, dataset=dataset, device=device)

    # Option 3: Generate single sample (fastest)
    # sample = sample_single_timeseries(dataset, device)

    print("Sampling functions ready! Use any of the three options above.")
"""