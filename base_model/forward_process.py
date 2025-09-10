import torch.nn.functional as F
import torch

# PRESERVED: Your exact comment style and content
# using linspace, create a list of timesteps which will be a schedule of noising process
# torch.linspace: Creates a one-dimensional tensor of size steps whose values are evenly spaced from start to end, inclusive
def linear_beta_schedule(timesteps, start=0.0001, end=0.02):
    """
    Create a linear schedule for the noise variance (beta) parameters
    
    The beta schedule controls how much noise is added at each timestep during the forward process.
    A linear schedule works well for most diffusion applications, providing smooth noise increase.
    
    Args:
        timesteps: Total number of diffusion timesteps (typically 300-1000)
        start: Starting noise level (small value to preserve signal early)
        end: Ending noise level (larger value for complete noise at end)
    
    Returns:
        torch.Tensor: Linear schedule of beta values from start to end
    """
    return torch.linspace(start, end, timesteps)

def get_index_from_list(vals, t, x_shape):
    """
    Extract specific values from a tensor based on timestep indices
    
    This utility function is crucial for diffusion models. It extracts the correct
    diffusion parameters (betas, alphas, etc.) for each sample in a batch based on
    their individual timesteps, then reshapes for proper broadcasting.
    
    Args:
        vals: Tensor containing precomputed values (e.g., betas, alphas_cumprod)
        t: Tensor of timestep indices for each sample in batch [batch_size]
        x_shape: Shape of the input data tensor (used for broadcasting)
    
    Returns:
        torch.Tensor: Selected values reshaped for broadcasting with input data
    """
    # PRESERVED: Your exact comments
    batch_size = t.shape[0]             # Get the batch size from the timestep tensor
    out = vals.gather(-1, t.cpu())      # Extract values at timestep indices t from the vals tensor
    # Reshape output to match input dimensions:
    # - First dimension: batch_size of timestep tensor
    # - Remaining dimensions: all 1s (for broadcasting)
    # Then move result back to original device of t
    return out.reshape(batch_size, *((1,) * (len(x_shape) - 1))).to(t.device)

def forward_diffusion_sample(x_0, t, device="cpu"):
    """
    Apply forward diffusion process: add noise to clean data
    
    This implements the key equation of DDPM forward process:
    q(x_t | x_0) = N(x_t; √ᾱ_t * x_0, (1 - ᾱ_t) * I)
    
    Using the reparameterization trick:
    x_t = √ᾱ_t * x_0 + √(1 - ᾱ_t) * ε
    
    Where:
    - x_0 is the original clean data
    - x_t is the noisy data at timestep t  
    - ᾱ_t is the cumulative product of alphas up to timestep t
    - ε is random Gaussian noise
    
    Args:
        x_0: Clean input data [batch, features, time]
        t: Timestep indices [batch] - which diffusion step to apply
        device: Device to run computation on (CPU/GPU)
    
    Returns:
        tuple: (noisy_data, noise) where:
            - noisy_data: x_t with appropriate amount of noise added
            - noise: the actual noise that was added (needed for training loss)
    """
    # PRESERVED: Your exact comments
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

# ============================================================================
# PRECOMPUTED DIFFUSION PARAMETERS (PRESERVED EXACTLY)
# ============================================================================

# PRESERVED: Your exact comment and value
# Define beta schedule
T = 300
betas = linear_beta_schedule(timesteps=T)

# ============================================================================
# PRE-CALCULATED TERMS FOR EFFICIENT COMPUTATION
# ============================================================================
# These values are computed once and reused throughout training for efficiency
# They implement the mathematical relationships derived in the DDPM paper

# PRESERVED: Your exact comments
# Pre-calculated different terms for closed form
# inverse relationship in DDPM
alphas = 1. - betas

# PRESERVED: Your exact comment
# Represents how much of the original signal remains after t steps
alphas_cumprod = torch.cumprod(alphas, axis=0)

# PRESERVED: Your exact multi-line comment block
# Creates [1.0, ᾱ₀, ᾱ₁, ᾱ₂, ...] by:
# - [:-1] removes last element: [ᾱ₀, ᾱ₁, ᾱ₂, ...]
# - F.pad adds 1.0 at beginning: [1.0, ᾱ₀, ᾱ₁, ᾱ₂, ...]
# - (1, 0) means pad 1 element on left, 0 on right
# Needed because at t=0, the "previous" cumulative product is 1.0 (no noise yet)
alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

# PRESERVED: Your exact comment
# sed in reverse process denoising step
sqrt_recip_alphas = torch.sqrt(1.0 / alphas)

# PRESERVED: Your exact comment
# Forward process coefficients
sqrt_alphas_cumprod = torch.sqrt(alphas_cumprod)
sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - alphas_cumprod)

# PRESERVED: Your exact multi-line comment block
# Most complex: Posterior variance for reverse process
# Formula: βₛ × (1 - ᾱₛ₋₁) / (1 - ᾱₛ)
# This is the optimal variance when denoising from xₛ to xₛ₋₁(from DDPM paper)
posterior_variance = betas * (1. - alphas_cumprod_prev) / (1. - alphas_cumprod)

# ============================================================================
# MATHEMATICAL INTUITION AND EXPLANATION
# ============================================================================
"""
DETAILED EXPLANATION OF DIFFUSION MATHEMATICS:

The forward diffusion process gradually corrupts data by adding Gaussian noise:
x₀ → x₁ → x₂ → ... → x_T (clean data to pure noise)

At each step t, we apply:
q(xₜ | xₜ₋₁) = N(xₜ; √(1-βₜ) * xₜ₋₁, βₜ * I)

KEY INSIGHT: Using the reparameterization trick, we can jump directly from x₀ to xₜ:
q(xₜ | x₀) = N(xₜ; √ᾱₜ * x₀, (1-ᾱₜ) * I)

Where ᾱₜ = ∏ᵢ₌₁ᵗ αᵢ = ∏ᵢ₌₁ᵗ (1-βᵢ)

PRECOMPUTED TERMS EXPLAINED:

1. betas (βₜ): Noise variance schedule
   - Controls how much noise to add at each step
   - Linear schedule: small initially, larger towards end
   - βₜ ∈ [0.0001, 0.02] for T=300 steps

2. alphas (αₜ): Signal retention coefficient
   - αₜ = 1 - βₜ
   - How much of previous signal to keep at step t
   - Decreases from ~0.9999 to ~0.98 over T steps

3. alphas_cumprod (ᾱₜ): Cumulative signal retention
   - ᾱₜ = ∏ᵢ₌₁ᵗ αᵢ
   - Total signal remaining after t steps
   - Decreases from 1.0 to near 0 over T steps

4. alphas_cumprod_prev (ᾱₜ₋₁): Previous cumulative retention
   - Needed for reverse process variance calculation
   - Padded with 1.0 at beginning (t=0 has no previous step)

5. sqrt_recip_alphas (1/√αₜ): Reverse process scaling
   - Used in denoising equation: x_{t-1} = (1/√αₜ) * (...)
   - Compensates for signal scaling during forward process

6. sqrt_alphas_cumprod (√ᾱₜ): Forward process signal coefficient
   - Multiplies original data in forward process
   - Controls how much original signal remains

7. sqrt_one_minus_alphas_cumprod (√(1-ᾱₜ)): Forward process noise coefficient
   - Multiplies noise in forward process  
   - Controls how much noise to add

8. posterior_variance: Optimal reverse process variance
   - σ²ₜ = βₜ * (1-ᾱₜ₋₁)/(1-ᾱₜ)
   - Derived from Bayes' rule for optimal denoising
   - Used in stochastic sampling (adds controlled noise back)

TRAINING EFFICIENCY:
- All terms precomputed once, reused throughout training
- Forward process: O(1) to jump to any timestep t
- No need to simulate entire Markov chain step by step
- Enables efficient parallel training across timesteps

REVERSE PROCESS:
The neural network learns the reverse process:
p_θ(x_{t-1} | xₑ) = N(x_{t-1}; μ_θ(xₑ,t), σ²ₜ*I)

Where μ_θ is predicted by the network using these precomputed coefficients.

WHY THIS WORKS:
- Forward process has known closed form (no learning needed)
- Reverse process learned by neural network (the hard part)
- Symmetric: if we can add noise optimally, we can remove it optimally
- Gradual: many small steps easier than one big step
- Stable: each step is well-conditioned Gaussian operation
"""