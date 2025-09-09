import torch
import matplotlib.pyplot as plt
import numpy as np
import config
from forward_process import (
    get_index_from_list, 
    sqrt_one_minus_alphas_cumprod, 
    sqrt_recip_alphas, 
    betas, 
    posterior_variance
)

# Your existing sampling function - NO CHANGES NEEDED!
@torch.no_grad()
def sample_timestep(x, t, model):
    """
    Calls the model to predict the noise in the time series and returns
    the denoised time series.
    Applies noise to this time series, if we are not in the last step yet.
    """
    betas_t = get_index_from_list(betas, t, x.shape)
    sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
        sqrt_one_minus_alphas_cumprod, t, x.shape
    )
    sqrt_recip_alphas_t = get_index_from_list(sqrt_recip_alphas, t, x.shape)

    # Call model (current time series - noise prediction)
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
    Plots a single generated time series sample.
    Handles potential differences in data shape and dataset structure.
    """
    # Ensure data_sample is on CPU and remove batch dimension if present
    if torch.is_tensor(data_sample):
        data_sample = data_sample.detach().cpu()

    if data_sample.dim() == 3:
        # Assume shape is [batch_size, features, time] or [batch_size, time, features]
        # Take the first sample and ensure it's [features, time]
        data_sample = data_sample.squeeze(0) # Remove batch dim: [features, time]
    elif data_sample.dim() != 2:
         raise ValueError(f"Expected data_sample to have 2 or 3 dimensions, but got {data_sample.dim()}")

    # Transpose if shape is [time, features] to [features, time]
    if data_sample.shape[0] != config.INPUT_DIM and data_sample.shape[1] == config.INPUT_DIM:
         data_sample = data_sample.transpose(0, 1)


    num_features = data_sample.shape[0]
    sequence_length = data_sample.shape[1]

    fig, axes = plt.subplots(num_features, 1, figsize=(15, 3 * num_features))
    if num_features == 1:
        axes = [axes] # Ensure axes is always a list

    # Denormalize if dataset provided and has denormalize method
    if dataset is not None and hasattr(dataset, 'denormalize'):
        try:
            # Denormalize expects [batch=1, features, time] or similar
            data_sample_denorm = dataset.denormalize(data_sample.unsqueeze(0))
            data_sample_denorm = data_sample_denorm.squeeze(0)
        except Exception as e:
            print(f"Warning: Could not denormalize data: {e}")
            data_sample_denorm = data_sample # Use original if denormalization fails
    else:
        data_sample_denorm = data_sample # Use original if no dataset or no denormalize

    for i in range(num_features):
        axes[i].plot(data_sample_denorm[i])

        # Get column name, handle missing or insufficient names
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
    Sample and plot time series generation process.
    Adapted from sample_plot_image() for time series data.
    """
    # Sample noise - CHANGE: Use time series dimensions instead of image
    sequence_length = config.SEQUENCE_LENGTH  # Your global sequence length
    input_dim = config.INPUT_DIM             # Your global input dimension

    # Start with pure noise: [batch=1, features, time]
    timeseries = torch.randn((1, input_dim, sequence_length), device=device)

    plt.figure(figsize=(20, 12))
    num_plots = 10
    stepsize = int(T/num_plots)

    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        timeseries = sample_timestep(timeseries, t, model)

        # Clamp to valid range
        timeseries = torch.clamp(timeseries, -1.0, 1.0)

        # Plot intermediate steps
        if i % stepsize == 0:
            plt.subplot(2, num_plots//2, int(i/stepsize)+1)
            # show_timeseries_sample expects [features, time] or [batch=1, features, time]
            show_timeseries_sample(timeseries.detach().cpu(), dataset,
                                 title=f"Step {T-i}")

    plt.tight_layout()
    plt.show()

    return timeseries

@torch.no_grad()
def sample_multiple_timeseries(model, num_samples=5, dataset=None, device='cpu', T=1000):
    """
    Generate multiple time series samples and plot them.
    """
    sequence_length = config.SEQUENCE_LENGTH
    input_dim = config.INPUT_DIM

    samples = []

    fig, axes = plt.subplots(input_dim, 1, figsize=(15, 3*input_dim))
    if input_dim == 1:
        axes = [axes]

    for sample_idx in range(num_samples):
        print(f"Generating sample {sample_idx + 1}/{num_samples}...")

        # Start with noise
        timeseries = torch.randn((1, input_dim, sequence_length), device=device)

        # Denoise step by step
        for i in range(0, T)[::-1]:
            t = torch.full((1,), i, device=device, dtype=torch.long)
            timeseries = sample_timestep(timeseries, t, model)
            timeseries = torch.clamp(timeseries, -1.0, 1.0)

        sample = timeseries.detach().cpu().squeeze(0) # Remove batch dim: [features, time]
        samples.append(sample)

        # Plot the generated sample
        for i in range(input_dim):
            # Denormalize if dataset provided
            if dataset is not None and hasattr(dataset, 'denormalize'):
                 try:
                     sample_denorm = dataset.denormalize(sample.unsqueeze(0)).squeeze(0)
                 except Exception as e:
                     print(f"Warning: Could not denormalize data for plotting: {e}")
                     sample_denorm = sample # Use original if denormalization fails
            else:
                 sample_denorm = sample # Use original if no dataset or no denormalize


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
    Generate a single time series sample (fastest option).
    """
    sequence_length = config.SEQUENCE_LENGTH
    input_dim = config.INPUT_DIM

    # Start with noise
    timeseries = torch.randn((1, input_dim, sequence_length), device=device)

    # Denoise step by step
    for i in range(0, T)[::-1]:
        t = torch.full((1,), i, device=device, dtype=torch.long)
        timeseries = sample_timestep(timeseries, t, model) # Pass the model here
        timeseries = torch.clamp(timeseries, -1.0, 1.0)

    # Plot the result
    # show_timeseries_sample expects [features, time] or [batch=1, features, time]
    show_timeseries_sample(timeseries.detach().cpu(), dataset, "Generated Time Series")

    return timeseries.detach().cpu()

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