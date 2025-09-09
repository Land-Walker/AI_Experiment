import torch
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.metrics import mean_squared_error, mean_absolute_error
from scipy.spatial.distance import euclidean
import pandas as pd

class DiffusionModelEvaluator:
    """
    Comprehensive evaluation framework for time series diffusion models.
    Computes CRPS, distance metrics, and distributional comparisons.
    """

    def __init__(self, model, dataset, device='cpu'):
        self.model = model
        self.dataset = dataset
        self.device = device
        self.model.eval()

    def crps_gaussian(self, observations, forecasts, std_forecasts):
        """
        Compute CRPS assuming Gaussian distribution for forecasts.

        Args:
            observations: Real values [n_samples]
            forecasts: Predicted means [n_samples]
            std_forecasts: Predicted standard deviations [n_samples]
        """
        # Standardize
        standardized = (observations - forecasts) / std_forecasts

        # CRPS formula for Gaussian distribution
        crps = std_forecasts * (
            standardized * (2 * stats.norm.cdf(standardized) - 1) +
            2 * stats.norm.pdf(standardized) - 1/np.sqrt(np.pi)
        )

        return crps

    def crps_empirical(self, observations, forecast_samples):
        """
        Compute empirical CRPS from multiple forecast samples.

        Args:
            observations: Real values [n_timesteps, n_features]
            forecast_samples: Generated samples [n_samples, n_timesteps, n_features]
        """
        n_samples, n_timesteps, n_features = forecast_samples.shape
        crps_values = np.zeros((n_timesteps, n_features))

        for t in range(n_timesteps):
            for f in range(n_features):
                obs_val = observations[t, f]
                samples = forecast_samples[:, t, f]

                # Sort samples
                samples_sorted = np.sort(samples)

                # Empirical CDF
                p_values = np.arange(1, n_samples + 1) / n_samples

                # CRPS calculation
                crps_sum = 0
                for i, sample in enumerate(samples_sorted):
                    p = p_values[i]
                    if obs_val <= sample:
                        crps_sum += (p**2) * abs(sample - obs_val)
                    else:
                        crps_sum += ((1-p)**2) * abs(sample - obs_val)

                # Add the integral of (F(x) - 1)^2 for x > obs
                crps_sum += np.sum(np.where(samples_sorted > obs_val,
                                          (1 - p_values)**2 * (samples_sorted - obs_val), 0))

                crps_values[t, f] = crps_sum / n_samples

        return crps_values

    def generate_samples(self, n_samples=100, sequence_length=None, T=1000):
        """Generate multiple samples from the diffusion model."""
        if sequence_length is None:
            sequence_length = self.dataset.sequence_length

        input_dim = self.dataset.input_dim
        samples = []

        print(f"Generating {n_samples} samples...")
        with torch.no_grad():
            for i in range(n_samples):
                if i % 10 == 0:
                    print(f"Sample {i+1}/{n_samples}")

                # Start with noise
                timeseries = torch.randn((1, input_dim, sequence_length), device=self.device)

                # Denoise step by step
                for step in range(T-1, -1, -1):
                    t = torch.full((1,), step, device=self.device, dtype=torch.long)
                    # Pass the model argument here
                    timeseries = sample_timestep(timeseries, t, self.model)
                    timeseries = torch.clamp(timeseries, -1.0, 1.0)

                # Convert to [time, features] and denormalize
                sample = timeseries[0].transpose(0, 1).cpu().numpy()  # [features, time] -> [time, features]

                if hasattr(self.dataset, 'denormalize'):
                    try:
                        sample = self.dataset.denormalize(sample.T).T  # Denormalize expects [features, time]
                    except:
                        pass  # Keep normalized if denormalization fails

                samples.append(sample)

        return np.array(samples)  # [n_samples, time, features]

    def get_real_data_samples(self, n_samples=None):
        """Get real data samples for comparison."""
        if n_samples is None:
            n_samples = min(100, len(self.dataset))

        real_samples = []
        indices = np.random.choice(len(self.dataset), size=n_samples, replace=False)

        for idx in indices:
            sample = self.dataset[idx]  # [features, time]
            sample = sample.transpose(0, 1).numpy()  # [time, features]

            if hasattr(self.dataset, 'denormalize'):
                try:
                    sample = self.dataset.denormalize(sample.T).T
                except:
                    pass

            real_samples.append(sample)

        return np.array(real_samples)  # [n_samples, time, features]

    def compute_distance_metrics(self, real_samples, synthetic_samples):
        """Compute various distance metrics between real and synthetic data."""
        metrics = {}

        # Flatten samples for some metrics
        real_flat = real_samples.reshape(real_samples.shape[0], -1)
        synthetic_flat = synthetic_samples.reshape(synthetic_samples.shape[0], -1)

        # 1. Mean Euclidean Distance between sample means
        real_mean = np.mean(real_samples, axis=0)  # [time, features]
        synthetic_mean = np.mean(synthetic_samples, axis=0)  # [time, features]
        metrics['euclidean_distance'] = euclidean(real_mean.flatten(), synthetic_mean.flatten())

        # 2. MSE and MAE between means
        metrics['mse_means'] = mean_squared_error(real_mean.flatten(), synthetic_mean.flatten())
        metrics['mae_means'] = mean_absolute_error(real_mean.flatten(), synthetic_mean.flatten())

        # 3. Standard deviation comparison
        real_std = np.std(real_samples, axis=0)
        synthetic_std = np.std(synthetic_samples, axis=0)
        metrics['std_mse'] = mean_squared_error(real_std.flatten(), synthetic_std.flatten())

        # 4. Correlation comparison (per feature)
        correlations = {}
        for f in range(real_samples.shape[2]):
            real_corr = np.corrcoef(real_samples[:, :, f])
            synthetic_corr = np.corrcoef(synthetic_samples[:, :, f])
            correlations[f] = mean_squared_error(real_corr.flatten(), synthetic_corr.flatten())
        metrics['correlation_mse'] = correlations

        # 5. Wasserstein distance (approximate)
        try:
            from scipy.stats import wasserstein_distance
            w_distances = []
            for f in range(real_samples.shape[2]):
                for t in range(real_samples.shape[1]):
                    wd = wasserstein_distance(real_samples[:, t, f], synthetic_samples[:, t, f])
                    w_distances.append(wd)
            metrics['wasserstein_distance_mean'] = np.mean(w_distances)
        except ImportError:
            metrics['wasserstein_distance_mean'] = None

        return metrics

    def evaluate_model(self, n_samples=100, T=1000, save_plots=True):
        """
        Complete evaluation of the diffusion model.

        Returns:
            dict: Comprehensive evaluation metrics
        """
        print("Starting model evaluation...")

        # Generate samples
        print("\n1. Generating synthetic samples...")
        synthetic_samples = self.generate_samples(n_samples=n_samples, T=T)

        print("\n2. Loading real data samples...")
        real_samples = self.get_real_data_samples(n_samples=n_samples)

        print(f"Real samples shape: {real_samples.shape}")
        print(f"Synthetic samples shape: {synthetic_samples.shape}")

        # Compute CRPS
        print("\n3. Computing CRPS...")
        # Use a subset of real data as "observations"
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

        # Compute distance metrics
        print("\n4. Computing distance metrics...")
        distance_metrics = self.compute_distance_metrics(real_samples, synthetic_samples)

        # Visualization
        if save_plots:
            print("\n5. Creating visualizations...")
            self.plot_comparison(real_samples, synthetic_samples, crps_mean)

        # Compile results
        results = {
            'crps_sum': crps_total,
            'crps_per_feature': np.sum(crps_mean, axis=0),
            'crps_per_timestep': np.sum(crps_mean, axis=1),
            'crps_mean_matrix': crps_mean,
            **distance_metrics,
            'n_samples': n_samples,
            'n_timesteps': synthetic_samples.shape[1],
            'n_features': synthetic_samples.shape[2]
        }

        # Print summary
        self.print_evaluation_summary(results)

        return results

    def plot_comparison(self, real_samples, synthetic_samples, crps_scores):
        """Create comparison plots between real and synthetic data."""
        n_features = real_samples.shape[2]
        feature_names = getattr(self.dataset, 'column_names', [f'Feature {i+1}' for i in range(n_features)])

        fig, axes = plt.subplots(2, n_features, figsize=(4*n_features, 8))
        if n_features == 1:
            axes = axes.reshape(2, 1)

        # Plot 1: Sample trajectories
        for f in range(n_features):
            ax = axes[0, f]

            # Plot some real samples
            for i in range(min(10, len(real_samples))):
                ax.plot(real_samples[i, :, f], alpha=0.3, color='blue', linewidth=0.5)

            # Plot some synthetic samples
            for i in range(min(10, len(synthetic_samples))):
                ax.plot(synthetic_samples[i, :, f], alpha=0.3, color='red', linewidth=0.5)

            # Plot means
            ax.plot(np.mean(real_samples, axis=0)[:, f], color='blue', linewidth=2, label='Real (mean)')
            ax.plot(np.mean(synthetic_samples, axis=0)[:, f], color='red', linewidth=2, label='Synthetic (mean)')

            ax.set_title(f'{feature_names[f]} - Sample Comparison')
            ax.legend()
            ax.grid(True)

        # Plot 2: CRPS scores over time
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
        """Print a summary of evaluation results."""
        print("\n" + "="*60)
        print("DIFFUSION MODEL EVALUATION SUMMARY")
        print("="*60)

        print(f"\n📊 Dataset Info:")
        print(f"   • Samples evaluated: {results['n_samples']}")
        print(f"   • Time steps: {results['n_timesteps']}")
        print(f"   • Features: {results['n_features']}")

        print(f"\n📈 CRPS Scores:")
        print(f"   • Total CRPS Sum: {results['crps_sum']:.4f}")
        print(f"   • Average CRPS per feature: {np.mean(results['crps_per_feature']):.4f}")

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

# Usage example
def evaluate_trained_model(model, dataset, device='cpu', n_samples=50, T=1000):
    """
    Convenience function to evaluate a trained diffusion model.

    Args:
        model: Trained diffusion model
        dataset: TimeSeriesDataset used for training
        device: Device to run evaluation on
        n_samples: Number of samples to generate for evaluation
        T: Number of diffusion steps

    Returns:
        dict: Evaluation results
    """
    evaluator = DiffusionModelEvaluator(model, dataset, device)
    results = evaluator.evaluate_model(n_samples=n_samples, T=T)
    return results

"""
Example usage:
if __name__ == "__main__":
    # Assuming you have a trained model and dataset
    # results = evaluate_trained_model(model, dataset, device='cpu', n_samples=50)
    print("Evaluation framework ready!")
    print("Usage: results = evaluate_trained_model(model, dataset, device='cpu')")
"""