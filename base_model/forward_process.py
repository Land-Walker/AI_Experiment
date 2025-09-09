import torch.nn.functional as F
import torch

# using linspace, create a list of timesteps which will be a schedule of noising process
# torch.linspace: Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
  return torch.linspace(start, end, timesteps)


def get_index_from_list(vals, t, x_shape):
  """
  Returns a specific index t of a passed list of values vals
  while considering the batch dimension.
  """
  batch_size = t.shape[0]             # Get the batch size from the timestep tensor
  out = vals.gather(-1, t.cpu())      # Extract values at timestep indices t from the vals tensor
  # Reshape output to match input dimensions:
  # - First dimension: batch_size of timestep tensor
  # - Remaining dimensions: all 1s (for broadcasting)
  # Then move result back to original device of t
  return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)


def forward_diffusion_sample(x_0, t, device="cpu"):
  """
  Takes data and a timestep as input and
  returns the noisy version of it
  """
  noise = torch.randn_like(x_0)         # Create a noise
  # Initialize variable that controls how much original image to keep (cumulative product of alphas upto timestep t)
  sqrt_alphas_cumprod_t = get_index_from_list(sqrt_alphas_cumprod, t, x_0.shape)
  # Get (1-alpha) which controls how much noise to add
  sqrt_one_minus_alphas_cumprod_t = get_index_from_list(
      sqrt_one_minus_alphas_cumprod, t, x_0.shape
  )
  # mean + variance & return both noisy data and the noise (required in training)
  return sqrt_alphas_cumprod_t.to(device) * x_0.to(device) \
    + sqrt_one_minus_alphas_cumprod_t.to(device) * noise.to(device), noise.to(device)


# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)


# Pre-calculated different terms for closed form
# inverse relationship in DDPM
alphas = 1. - betas
# Represents how much of the original signal remains after t steps
alphas_cumprod = torch.cumprod(alphas, axis=0)
# Creates [1.0, ᾱ₀, ᾱ₁, ᾱ₂, ...] by:
# - [:-1] removes last element: [ᾱ₀, ᾱ₁, ᾱ₂, ...]
# - F.pad adds 1.0 at beginning: [1.0, ᾱ₀, ᾱ₁, ᾱ₂, ...]
# - (1, 0) means pad 1 element on left, 0 on right
# Needed because at t=0, the "previous" cumulative product is 1.0 (no noise yet)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)
# sed in reverse process denoising step
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)
# Forward process coefficients
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)
# Most complex: Posterior variance for reverse process
# Formula: βₜ × (1 - ᾱₜ₋₁) / (1 - ᾱₜ)
# This is the optimal variance when denoising from xₜ to xₜ₋₁(from DDPM paper)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)