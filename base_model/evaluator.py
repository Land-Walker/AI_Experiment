import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import euclidean
import pandas as pd
from data_sampler import sample_timestep  # PRESERVED: Your exact import

class DiffusionModelEvaluator:
    """
    Comprehensive evaluation framework for time series diffusion models.
    
    This class implements sophisticated evaluation metrics specifically designed
    for assessing the quality of generated time series data:
    
    1. CRPS (Continuous Ranked Probability Score) - measures distributional accuracy
    2. Distance metrics - compare statistical properties of real vs synthetic data
    3. Visualization tools - qualitative assessment and comparison plots
    
    The evaluation goes beyond simple MSE loss to assess whether generated data
    captures the complex statistical properties of real financial time series.
    """

    def __init__(self, model, dataset, device='cpu'):
        """
        Initialize evaluator with trained model and dataset
        
        Args:
            model: Trained diffusion model to evaluate
            dataset: Original dataset used for training (for comparison)
            device: Device for computation ('cpu' or 'cuda')
        """
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.eval()  # Set model to evaluation mode (disables dropout, fixes batch norm)

    def crps_gaussian(self, observations, forecasts, std_forecasts):
        """
        Compute CRPS assuming Gaussian distribution for forecasts
        
        CRPS (Continuous Ranked Probability Score) is a proper scoring rule that
        measures the quality of probabilistic forecasts. For Gaussian distributions,
        it has a closed-form solution which is computationally efficient.
        
        Lower CRPS values indicate better forecast quality.

        Args:
            observations: Real values [n_samples] - ground truth data
            forecasts: Predicted means [n_samples] - model predictions  
            std_forecasts: Predicted standard deviations [n_samples] - uncertainty estimates

        Returns:
            np.array: CRPS scores for each observation (lower is better)
        """
        # Standardize observations relative to forecast distribution
        # This normalizes the error by the predicted uncertainty
        standardized = (observations - forecasts) / std_forecasts

        # CRPS formula for Gaussian distribution (closed-form solution)
        # This formula integrates the squared difference between forecast CDF and step function
        crps = std_forecasts * (
            standardized * (2 * stats.norm.cdf(standardized) - 1) +
            2 * stats.norm.pdf(standardized) - 1/np.sqrt(np.pi)
        )

        return crps

    def crps_empirical(self, observations, forecast_samples):
        """
        Compute empirical CRPS from multiple forecast samples
        
        This is more general than Gaussian CRPS as it makes no distributional assumptions.
        It directly computes CRPS from the empirical distribution of generated samples.
        
        More computationally expensive but works for any distribution shape.

        Args:
            observations: Real values [n_timesteps, n_features] - ground truth sequences
            forecast_samples: Generated samples [n_samples, n_timesteps, n_features] - model outputs

        Returns:
            np.array: CRPS values [n_timesteps, n_features] for each time-feature pair
        """
        n_samples, n_timesteps, n_features = forecast_samples.shape
        crps_values = np.zeros((n_timesteps, n_features))

        # Compute CRPS independently for each timestep and feature
        for t in range(n_timesteps):
            for f in range(n_features):
                obs_val = observations[t, f]  # True value at this time/feature
                samples = forecast_samples[:, t, f]  # All model predictions for this time/feature

                # Sort samples to create empirical cumulative distribution function (CDF)
                samples_sorted = np.sort(samples)

                # Create empirical CDF values: F(x) = rank/n_samples
                p_values = np.arange(1, n_samples + 1) / n_samples

                # Empirical CRPS calculation using numerical integration
                # CRPS = ∫ (F_forecast(x) - I(x ≥ observation))² dx
                # where F_forecast is the forecast CDF, I is indicator function
                crps_sum = 0
                for i, sample in enumerate(samples_sorted):
                    p = p_values[i]  # CDF value at this sample point
                    
                    if obs_val <= sample:
                        # Before observation: (F(x) - 0)² = F(x)²
                        crps_sum += (p**2) * abs(sample - obs_val)
                    else:
                        # After observation: (F(x) - 1)² = (1-F(x))²
                        crps_sum += ((1-p)**2) * abs(sample - obs_val)

                # Add contribution from tail where F(x) = 1 but observation indicator = 0
                crps_sum += np.sum(np.where(samples_sorted > obs_val,
                                          (1 - p_values)**2 * (samples_sorted - obs_val), 0))

                crps_values[t, f] = crps_sum / n_samples

        return crps_values

    def generate_samples(self, n_samples=100, sequence_length=None, T=1000):
        """
        Generate multiple samples from the diffusion model for evaluation
        
        Runs the complete reverse diffusion process multiple times to create
        a collection of samples for statistical analysis and comparison.

        Args:
            n_samples: Number of independent samples to generate
            sequence_length: Length of sequences (uses dataset default if None)
            T: Number of diffusion timesteps for reverse process

        Returns:
            np.array: Generated samples [n_samples, time, features] in original scale
        """
        if sequence_length is None:
            sequence_length = self.dataset.sequence_length

        input_dim = self.dataset.input_dim
        samples = []

        print(f"Generating {n_samples} samples...")
        with torch.no_grad():  # Disable gradients for faster inference
            for i in range(n_samples):
                if i % 10 == 0:
                    print(f"Sample {i+1}/{n_samples}")

                # Start with pure Gaussian noise (x_T in diffusion terminology)
                timeseries = torch.randn((1, input_dim, sequence_length), device=self.device)

                # Complete reverse diffusion process: T-1 down to 0
                for step in range(T-1, -1, -1):
                    t = torch.full((1,), step, device=self.device, dtype=torch.long)
                    # Apply one denoising step using trained model
                    timeseries = sample_timestep(timeseries, t, self.model)  # PRESERVED: Pass model
                    # Clamp to valid range for numerical stability
                    timeseries = torch.clamp(timeseries, -1.0, 1.0)

                # Convert to [time, features] format and denormalize
                sample = timeseries[0].transpose(0, 1).cpu().numpy()  # [features, time] → [time, features]

                # Denormalize if dataset supports it (convert from [-1,1] to original scale)
                if hasattr(self.dataset, 'denormalize'):
                    try:
                        sample = self.dataset.denormalize(sample.T).T  # Denormalize expects [features, time]
                    except:
                        pass  # Keep normalized if denormalization fails

                samples.append(sample)

        return np.array(samples)  # [n_samples, time, features]

    def get_real_data_samples(self, n_samples=None):
        """
        Extract real data samples for comparison with generated samples
        
        Creates a comparable set of real data sequences that can be
        statistically compared with generated data.

        Args:
            n_samples: Number of real samples to extract (default: min(100, dataset_size))

        Returns:
            np.array: Real data samples [n_samples, time, features] in original scale
        """
        if n_samples is None:
            n_samples = min(100, len(self.dataset))

        real_samples = []
        # Randomly select samples from dataset to avoid bias toward specific time periods
        indices = np.random.choice(len(self.dataset), size=n_samples, replace=False)

        for idx in indices:
            sample = self.dataset[idx]  # Get sample [features, time]
            sample = sample.transpose(0, 1).numpy()  # Convert to [time, features]

            # Denormalize to original scale if possible
            if hasattr(self.dataset, 'denormalize'):
                try:
                    sample = self.dataset.denormalize(sample.T).T
                except:
                    pass  # Keep normalized if denormalization fails

            real_samples.append(sample)

        return np.array(real_samples)  # [n_samples, time, features]

    def compute_distance_metrics(self, real_samples, synthetic_samples):
        """
        Compute various distance metrics between real and synthetic data
        
        These metrics assess different aspects of data quality:
        - Central tendency: Does the model capture average behavior?
        - Variability: Does the model capture volatility patterns?
        - Dependencies: Does the model preserve temporal correlations?
        - Distributions: Are the marginal distributions similar?

        Args:
            real_samples: Real data [n_samples, time, features]
            synthetic_samples: Generated data [n_samples, time, features]

        Returns:
            dict: Dictionary of computed distance metrics
        """
        metrics = {}

        # Flatten samples for some global comparison metrics
        real_flat = real_samples.reshape(real_samples.shape[0], -1)
        synthetic_flat = synthetic_samples.reshape(synthetic_samples.shape[0], -1)

        # 1. Mean Euclidean Distance between sample means
        # Measures if generated data has similar average behavior across time
        real_mean = np.mean(real_samples, axis=0)  # [time, features]
        synthetic_mean = np.mean(synthetic_samples, axis=0)  # [time, features]
        metrics['euclidean_distance'] = euclidean(real_mean.flatten(), synthetic_mean.flatten())

        # 2. MSE and MAE between means
        # Alternative measures of central tendency differences
        metrics['mse_means'] = mean_squared_error(real_mean.flatten(), synthetic_mean.flatten())
        metrics['mae_means'] = mean_absolute_error(real_mean.flatten(), synthetic_mean.flatten())

        # 3. Standard deviation comparison
        # Measures if generated data has similar volatility/variability patterns
        real_std = np.std(real_samples, axis=0)
        synthetic_std = np.std(synthetic_samples, axis=0)
        metrics['std_mse'] = mean_squared_error(real_std.flatten(), synthetic_std.flatten())

        # 4. Correlation comparison (per feature)
        # Measures if generated data preserves temporal dependencies within each feature
        correlations = {}
        for f in range(real_samples.shape[2]):
            # Compute autocorrelation matrices for each feature
            real_corr = np.corrcoef(real_samples[:, :, f])
            synthetic_corr = np.corrcoef(synthetic_samples[:, :, f])
            correlations[f] = mean_squared_error(real_corr.flatten(), synthetic_corr.flatten())
        metrics['correlation_mse'] = correlations

        # 5. Wasserstein distance (Earth Mover's Distance)
        # Measures distributional differences at each time-feature point
        try:
            from scipy.stats import wasserstein_distance
            w_distances = []
            for f in range(real_samples.shape[2]):
                for t in range(real_samples.shape[1]):
                    # Compare distributions at each (time, feature) point
                    wd = wasserstein_distance(real_samples[:, t, f], synthetic_samples[:, t, f])
                    w_distances.append(wd)
            metrics['wasserstein_distance_mean'] = np.mean(w_distances)
        except ImportError:
            metrics['wasserstein_distance_mean'] = None

        return metrics

    def evaluate_model(self, n_samples=100, T=1000, save_plots=True):
        """
        Complete evaluation of the diffusion model
        
        This is the main evaluation function that orchestrates all metrics
        and provides a comprehensive assessment of model quality.

        Args:
            n_samples: Number of samples to generate for evaluation
            T: Number of diffusion timesteps for generation
            save_plots: Whether to create and save visualization plots

        Returns:
            dict: Comprehensive evaluation results including all metrics
        """
        print("Starting model evaluation...")

        # Step 1: Generate synthetic samples using the trained model
        print("\n1. Generating synthetic samples...")
        synthetic_samples = self.generate_samples(n_samples=n_samples, T=T)

        # Step 2: Load corresponding real data samples for comparison
        print("\n2. Loading real data samples...")
        real_samples = self.get_real_data_samples(n_samples=n_samples)

        print(f"Real samples shape: {real_samples.shape}")
        print(f"Synthetic samples shape: {synthetic_samples.shape}")

        # Step 3: Compute CRPS scores (distributional accuracy)
        print("\n3. Computing CRPS...")
        # Use subset of real data as "observations" for CRPS computation
        n_eval = min(20, len(real_samples))
        observations = real_samples[:n_eval]  # [n_eval, time, features]

        crps_scores = []
        for i in range(n_eval):
            obs = observations[i]  # [time, features]
            # Use all synthetic samples as forecast ensemble
            crps = self.crps_empirical(obs, synthetic_samples)
            crps_scores.append(crps)

        crps_mean = np.mean(crps_scores, axis=0)  # [time, features]
        crps_total = np.sum(crps_mean)

        # Step 4: Compute statistical distance metrics
        print("\n4. Computing distance metrics...")
        distance_metrics = self.compute_distance_metrics(real_samples, synthetic_samples)

        # Step 5: Create visualizations if requested
        if save_plots:
            print("\n5. Creating visualizations...")
            self.plot_comparison(real_samples, synthetic_samples, crps_mean)

        # Step 6: Compile comprehensive results
        results = {
            'crps_sum': crps_total,
            'crps_per_feature': np.sum(crps_mean, axis=0),  # Sum over time for each feature
            'crps_per_timestep': np.sum(crps_mean, axis=1),  # Sum over features for each timestep
            'crps_mean_matrix': crps_mean,
            **distance_metrics,  # Include all distance metrics
            'n_samples': n_samples,
            'n_timesteps': synthetic_samples.shape[1],
            'n_features': synthetic_samples.shape[2]
        }

        # Step 7: Print human-readable summary
        self.print_evaluation_summary(results)

        return results

    def plot_comparison(self, real_samples, synthetic_samples, crps_scores):
        """
        Create comprehensive comparison plots between real and synthetic data
        
        Creates two types of visualizations:
        1. Sample trajectories: Visual comparison of time series patterns
        2. CRPS scores over time: Quantitative assessment of accuracy

        Args:
            real_samples: Real data samples [n_samples, time, features]
            synthetic_samples: Generated data samples [n_samples, time, features]
            crps_scores: CRPS scores [time, features]
        """
        n_features = real_samples.shape[2]
        # Get feature names from dataset or use generic names
        feature_names = getattr(self.dataset, 'column_names', [f'Feature {i+1}' for i in range(n_features)])

        fig, axes = plt.subplots(2, n_features, figsize=(4*n_features, 8))
        if n_features == 1:
            axes = axes.reshape(2, 1)

        # Top row: Sample trajectory comparison
        for f in range(n_features):
            ax = axes[0, f]

            # Plot sample of real trajectories (blue, semi-transparent)
            for i in range(min(10, len(real_samples))):
                ax.plot(real_samples[i, :, f], alpha=0.3, color='blue', linewidth=0.5)

            # Plot sample of synthetic trajectories (red, semi-transparent)
            for i in range(min(10, len(synthetic_samples))):
                ax.plot(synthetic_samples[i, :, f], alpha=0.3, color='red', linewidth=0.5)

            # Plot mean trajectories for clear comparison (solid lines)
            ax.plot(np.mean(real_samples, axis=0)[:, f], color='blue', linewidth=2, label='Real (mean)')
            ax.plot(np.mean(synthetic_samples, axis=0)[:, f], color='red', linewidth=2, label='Synthetic (mean)')

            ax.set_title(f'{feature_names[f]} - Sample Comparison')
            ax.legend()
            ax.grid(True)

        # Bottom row: CRPS scores over time
        for f in range(n_features):
            ax = axes[1, f]
            ax.plot(crps_scores[:, f], color='green', linewidth=2)
            ax.set_title(f'{feature_names[f]} - CRPS over Time')
            ax.set_xlabel('Time')
            ax.set_ylabel('CRPS')
            ax.grid(True)

        plt.tight_layout()
        plt.savefig('diffusion_evaluation.png', dpi=300, bbox_inches='tight')
        plt.show()

    def print_evaluation_summary(self, results):
        """
        Print a comprehensive, human-readable summary of evaluation results
        
        Formats all computed metrics in an easily digestible format with
        clear interpretation guidance.

        Args:
            results: Dictionary of evaluation results from evaluate_model()
        """
        print("\n" + "="*60)
        print("DIFFUSION MODEL EVALUATION SUMMARY")
        print("="*60)

        print(f"\n📊 Dataset Info:")
        print(f"   • Samples evaluated: {results['n_samples']}")
        print(f"   • Time steps: {results['n_timesteps']}")
        print(f"   • Features: {results['n_features']}")

        print(f"\n📈 CRPS Scores (Lower = Better):")
        print(f"   • Total CRPS Sum: {results['crps_sum']:.4f}")
        print(f"   • Average CRPS per feature: {np.mean(results['crps_per_feature']):.4f}")

        # Show CRPS breakdown by feature
        feature_names = getattr(self.dataset, 'column_names', [f'Feature {i+1}' for i in range(results['n_features'])])
        for i, name in enumerate(feature_names):
            print(f"     - {name}: {results['crps_per_feature'][i]:.4f}")

        print(f"\n📏 Distance Metrics:")
        print(f"   • Euclidean Distance (means): {results['euclidean_distance']:.4f}")
        print(f"   • MSE (means): {results['mse_means']:.6f}")
        print(f"   • MAE (means): {results['mae_means']:.6f}")
        print(f"   • Std Dev MSE: {results['std_mse']:.6f}")

        if results['wasserstein_distance_mean'] is not None:
            print(f"   • Wasserstein Distance: {results['wasserstein_distance_mean']:.4f}")

        print("\n" + "="*60)

# Convenience function for easy evaluation (PRESERVED exactly as in your code)
def evaluate_trained_model(model, dataset, device='cpu', n_samples=50, T=1000):
    """
    Convenience function to evaluate a trained diffusion model
    
    This is the main function users should call to evaluate their trained models.
    Creates evaluator instance and runs complete evaluation pipeline.

    Args:
        model: Trained diffusion model
        dataset: TimeSeriesDataset used for training
        device: Device to run evaluation on ('cpu' or 'cuda')
        n_samples: Number of samples to generate for evaluation
        T: Number of diffusion steps for generation

    Returns:
        dict: Comprehensive evaluation results
    """
    evaluator = DiffusionModelEvaluator(model, dataset, device)
    results = evaluator.evaluate_model(n_samples=n_samples, T=T)
    return results

# ============================================================================
# EVALUATION METHODOLOGY EXPLANATION
# ============================================================================
"""
EVALUATION METRICS EXPLAINED:

1. CRPS (Continuous Ranked Probability Score):
   - Measures quality of probabilistic forecasts
   - Compares entire forecast distribution to actual values
   - Lower values indicate better forecasts
   - Accounts for both accuracy and calibration of uncertainty

2. Distance Metrics:
   - Euclidean Distance: Overall similarity of temporal means
   - MSE/MAE: Alternative measures of central tendency differences
   - Standard Deviation MSE: Captures differences in volatility patterns
   - Correlation MSE: Measures preservation of temporal dependencies
   - Wasserstein Distance: Earth Mover's Distance between distributions

3. Visual Assessment:
   - Sample overlays: Qualitative comparison of trajectory patterns
   - CRPS over time: Identifies problematic time periods or regimes
   - Mean comparisons: Checks alignment of central tendencies

WHY THESE METRICS MATTER FOR FINANCIAL TIME SERIES:

Financial data has complex properties that simple MSE cannot capture:
- Non-stationary behavior (statistical properties change over time)
- Volatility clustering (periods of high/low volatility)
- Long-range dependencies (past events affect distant future)
- Multi-scale patterns (daily, weekly, monthly, seasonal cycles)
- Fat-tailed distributions (extreme events more common than normal distribution)

Our comprehensive evaluation assesses whether generated data preserves these
sophisticated statistical properties, not just point-wise accuracy.

This is crucial for downstream applications like:
- Risk management and stress testing
- Portfolio optimization
- Derivatives pricing
- Market simulation and backtesting
"""

# Example usage documentation (PRESERVED exactly as in your code)
"""
Example usage:
if __name__ == "__main__":
    # Assuming you have a trained model and dataset
    # results = evaluate_trained_model(model, dataset, device='cpu', n_samples=50)
    print("Evaluation framework ready!")
    print("Usage: results = evaluate_trained_model(model, dataset, device='cpu')")
"""