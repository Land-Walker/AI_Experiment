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

# GluonTS imports for enhanced time series evaluation
try:
    from gluonts.dataset.common import ListDataset
    from gluonts.dataset.field_names import FieldName
    from gluonts.evaluation import Evaluator
    from gluonts.model.forecast import SampleForecast
    from gluonts.evaluation.backtest import make_evaluation_predictions
    from gluonts.dataset.split import split
    GLUONTS_AVAILABLE = True
    print("GluonTS integration enabled for enhanced evaluation")
except ImportError:
    GLUONTS_AVAILABLE = False
    print("Warning: GluonTS not available. Install with: pip install gluonts for enhanced evaluation features")

# ============================================================================
# ORIGINAL SAMPLING FUNCTIONS (UNCHANGED)
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
# GLUONTS INTEGRATION (NEW FEATURES)
# ============================================================================

class DiffusionGluonTSPredictor:
    """
    GluonTS-compatible predictor wrapper for your diffusion model.
    Allows professional evaluation using GluonTS metrics.
    """
    
    def __init__(self, diffusion_model, dataset, device='cpu', T=1000, prediction_length=30):
        if not GLUONTS_AVAILABLE:
            raise ImportError("GluonTS required for this predictor")
            
        self.model = diffusion_model
        self.dataset = dataset  
        self.device = device
        self.T = T
        self.prediction_length = prediction_length
        self.freq = getattr(dataset, 'freq', 'D')
        
        self.model.to(device)
        self.model.eval()
    
    def predict(self, gluonts_dataset, num_samples=100):
        """Generate forecasts compatible with GluonTS evaluation."""
        for data_entry in gluonts_dataset:
            target = data_entry[FieldName.TARGET]
            start_date = data_entry[FieldName.START] 
            item_id = data_entry.get(FieldName.ITEM_ID, "series")
            
            # Generate multiple forecast samples
            forecast_samples = []
            for i in range(num_samples):
                sample = self._generate_single_forecast(target)
                forecast_samples.append(sample)
            
            samples_array = np.array(forecast_samples)
            
            # Create GluonTS forecast object
            forecast = SampleForecast(
                samples=samples_array,
                start_date=pd.Period(start_date, freq=self.freq) + len(target),
                freq=self.freq,
                item_id=item_id
            )
            
            yield forecast
    
    def _generate_single_forecast(self, historical_data):
        """Generate one forecast using your diffusion model."""
        context_len = min(config.SEQUENCE_LENGTH, len(historical_data))
        context = historical_data[-context_len:]
        
        if len(context.shape) == 1:
            context = context.reshape(-1, 1)
        context = context.T  # [features, time]
        
        # Normalize if dataset supports it
        if hasattr(self.dataset, 'data_min') and self.dataset.data_min is not None:
            data_min = self.dataset.data_min[:, np.newaxis]
            data_max = self.dataset.data_max[:, np.newaxis]  
            context = 2 * (context - data_min) / (data_max - data_min) - 1
        
        # Generate forecast using your sampling
        with torch.no_grad():
            forecast_tensor = torch.randn(
                (1, context.shape[0], self.prediction_length), 
                device=self.device
            )
            
            for step in range(self.T-1, -1, -1):
                t = torch.full((1,), step, device=self.device, dtype=torch.long)
                forecast_tensor = sample_timestep(forecast_tensor, t, self.model)
                forecast_tensor = torch.clamp(forecast_tensor, -1.0, 1.0)
            
            forecast_np = forecast_tensor[0].cpu().numpy().T  # [pred_len, features]
            
            # Denormalize
            if hasattr(self.dataset, 'denormalize'):
                try:
                    forecast_np = self.dataset.denormalize(forecast_np.T).T
                except:
                    pass
                    
            return forecast_np

def convert_to_gluonts_dataset(data, start_date="2020-01-01", freq="D"):
    """Convert your data to GluonTS format for evaluation."""
    if not GLUONTS_AVAILABLE:
        return None
        
    if isinstance(data, pd.DataFrame):
        target_data = data.values
        if hasattr(data.index, 'to_pydatetime'):
            start_date = data.index[0]
    else:
        target_data = data
        
    if len(target_data.shape) == 1:
        target_data = target_data.reshape(-1, 1)
    
    gluonts_entry = {
        FieldName.TARGET: target_data,
        FieldName.START: pd.Timestamp(start_date),
        FieldName.ITEM_ID: "diffusion_series"
    }
    
    return ListDataset([gluonts_entry], freq=freq)

def evaluate_with_gluonts_metrics(model, dataset, prediction_length=30, num_samples=100):
    """
    Professional evaluation using GluonTS while keeping your model unchanged.
    """
    if not GLUONTS_AVAILABLE:
        print("GluonTS not available - skipping professional evaluation")
        return {}
    
    print(f"Running GluonTS evaluation with {num_samples} samples...")
    
    # Get data for evaluation
    if hasattr(dataset, 'raw_data'):
        eval_data = dataset.raw_data
    else:
        eval_data = dataset.data
        if hasattr(dataset, 'denormalize'):
            try:
                eval_data = dataset.denormalize(eval_data.T).T
            except:
                pass
    
    # Convert to GluonTS format
    gluonts_data = convert_to_gluonts_dataset(eval_data, freq=getattr(dataset, 'freq', 'D'))
    
    # Create predictor wrapper
    predictor = DiffusionGluonTSPredictor(
        diffusion_model=model,
        dataset=dataset,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        prediction_length=prediction_length
    )
    
    # Split data for evaluation
    _, test_template = split(gluonts_data, offset=-prediction_length)
    
    # Generate forecasts and get ground truth
    forecast_it, ts_it = make_evaluation_predictions(
        dataset=test_template,
        predictor=predictor, 
        num_samples=num_samples
    )
    
    forecasts = list(forecast_it)
    ground_truth = list(ts_it)
    
    # Evaluate with GluonTS metrics
    evaluator = Evaluator(quantiles=[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    agg_metrics, item_metrics = evaluator(iter(ground_truth), iter(forecasts))
    
    print("\nGluonTS Professional Evaluation Results:")
    print("=" * 60)
    print(f"{'Metric':<25} {'Value':<15}")
    print("-" * 40)
    
    key_metrics = ['MASE', 'sMAPE', 'MSIS', 'QuantileLoss[0.5]', 'Coverage[0.5]']
    for metric in key_metrics:
        if metric in agg_metrics:
            print(f"{metric:<25} {agg_metrics[metric]:<15.4f}")
    
    return {
        'gluonts_metrics': agg_metrics,
        'item_metrics': item_metrics,
        'forecasts': forecasts,
        'ground_truth': ground_truth
    }

def sample_with_gluonts_forecast(model, dataset, prediction_length=30, device='cpu', T=1000):
    """
    Generate forecasts using GluonTS-style prediction with uncertainty quantification.
    """
    if not GLUONTS_AVAILABLE:
        print("GluonTS not available - using basic forecasting")
        return _basic_forecast_fallback(model, dataset, prediction_length, device, T)
    
    print(f"Generating GluonTS-style forecast for {prediction_length} steps...")
    
    # Get historical data
    if hasattr(dataset, 'raw_data') and dataset.raw_data is not None:
        historical_data = dataset.raw_data
    else:
        historical_data = dataset.data
        if hasattr(dataset, 'denormalize'):
            try:
                historical_data = dataset.denormalize(historical_data.T).T
            except:
                pass
    
    # Create GluonTS dataset and predictor
    gluonts_data = convert_to_gluonts_dataset(
        historical_data, 
        start_date="2020-01-01",
        freq=getattr(dataset, 'freq', 'D')
    )
    
    predictor = DiffusionGluonTSPredictor(
        diffusion_model=model,
        dataset=dataset,
        device=device,
        T=T,
        prediction_length=prediction_length
    )
    
    # Generate forecasts
    num_samples = 100
    forecasts = list(predictor.predict(gluonts_data, num_samples=num_samples))
    
    if not forecasts:
        return {"error": "No forecasts generated"}
    
    forecast = forecasts[0]
    forecast_samples = forecast.samples  # [num_samples, pred_length, features]
    forecast_mean = np.mean(forecast_samples, axis=0)
    forecast_std = np.std(forecast_samples, axis=0)
    
    # Calculate quantiles for uncertainty
    quantiles = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    forecast_quantiles = {}
    for q in quantiles:
        forecast_quantiles[f'quantile_{q}'] = np.quantile(forecast_samples, q, axis=0)
    
    results = {
        'forecast_mean': forecast_mean,
        'forecast_std': forecast_std,
        'forecast_quantiles': forecast_quantiles,
        'forecast_samples': forecast_samples,
        'prediction_length': prediction_length,
        'num_samples': num_samples,
        'start_date': forecast.start_date,
        'freq': forecast.freq
    }
    
    # Visualize forecast
    _visualize_gluonts_forecast(historical_data, results, dataset)
    
    return results

def _basic_forecast_fallback(model, dataset, prediction_length, device, T):
    """Fallback forecasting when GluonTS is not available."""
    print("Using basic forecasting (GluonTS unavailable)")
    
    samples = []
    for i in range(10):
        timeseries = torch.randn((1, config.INPUT_DIM, prediction_length), device=device)
        
        for step in range(T-1, -1, -1):
            t = torch.full((1,), step, device=device, dtype=torch.long)
            timeseries = sample_timestep(timeseries, t, model)
            timeseries = torch.clamp(timeseries, -1.0, 1.0)
        
        sample = timeseries[0].cpu().numpy().T
        
        if hasattr(dataset, 'denormalize'):
            try:
                sample = dataset.denormalize(sample.T).T
            except:
                pass
                
        samples.append(sample)
    
    samples_array = np.array(samples)
    
    return {
        'forecast_mean': np.mean(samples_array, axis=0),
        'forecast_std': np.std(samples_array, axis=0),
        'forecast_samples': samples_array,
        'prediction_length': prediction_length,
        'method': 'basic_fallback'
    }

def _visualize_gluonts_forecast(historical_data, forecast_results, dataset):
    """Visualize GluonTS-style forecasts with uncertainty bands."""
    forecast_mean = forecast_results['forecast_mean']
    forecast_quantiles = forecast_results['forecast_quantiles']
    
    num_features = historical_data.shape[1] if len(historical_data.shape) > 1 else 1
    feature_names = getattr(dataset, 'column_names', [f'Feature {i+1}' for i in range(num_features)])
    
    hist_length = len(historical_data)
    pred_length = forecast_results['prediction_length']
    
    hist_time = range(hist_length)
    forecast_time = range(hist_length, hist_length + pred_length)
    
    fig, axes = plt.subplots(num_features, 1, figsize=(15, 4*num_features), sharex=True)
    if num_features == 1:
        axes = [axes]
    
    for feat_idx in range(num_features):
        ax = axes[feat_idx]
        
        # Historical data
        hist_values = historical_data[:, feat_idx] if num_features > 1 else historical_data
        ax.plot(hist_time, hist_values, color='blue', linewidth=2, label='Historical', alpha=0.8)
        
        # Forecast mean
        forecast_values = forecast_mean[:, feat_idx] if num_features > 1 else forecast_mean
        ax.plot(forecast_time, forecast_values, color='red', linewidth=2, label='Forecast Mean')
        
        # Uncertainty bands
        lower_80 = forecast_quantiles['quantile_0.1'][:, feat_idx] if num_features > 1 else forecast_quantiles['quantile_0.1']
        upper_80 = forecast_quantiles['quantile_0.9'][:, feat_idx] if num_features > 1 else forecast_quantiles['quantile_0.9']
        lower_50 = forecast_quantiles['quantile_0.25'][:, feat_idx] if num_features > 1 else forecast_quantiles['quantile_0.25']
        upper_50 = forecast_quantiles['quantile_0.75'][:, feat_idx] if num_features > 1 else forecast_quantiles['quantile_0.75']
        
        ax.fill_between(forecast_time, lower_80, upper_80, alpha=0.2, color='red', label='80% Prediction Interval')
        ax.fill_between(forecast_time, lower_50, upper_50, alpha=0.3, color='red', label='50% Prediction Interval')
        
        ax.axvline(x=hist_length, color='gray', linestyle='--', alpha=0.7)
        
        ax.set_title(f'Forecast: {feature_names[feat_idx]}')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.suptitle('GluonTS-Style Diffusion Model Forecast', fontsize=16)
    plt.tight_layout()
    plt.show()

# ============================================================================
# EXAMPLE USAGE
# ============================================================================

"""
Example usage of your enhanced data_sampler.py:

# 1. Original functionality (completely unchanged):
sample = sample_single_timeseries(model, dataset, device)
samples = sample_multiple_timeseries(model, num_samples=5, dataset=dataset)
generation_process = sample_plot_timeseries(model, dataset, device)

# 2. New GluonTS professional evaluation:
gluonts_metrics = evaluate_with_gluonts_metrics(model, dataset, prediction_length=30)
print("MASE score:", gluonts_metrics['gluonts_metrics']['MASE'])

# 3. Professional forecasting with uncertainty:
forecast_result = sample_with_gluonts_forecast(
    model=model,
    dataset=dataset, 
    prediction_length=30,
    device=device
)
print("Forecast generated with uncertainty quantification")

Key benefits:
- All your existing code works exactly the same
- Professional time series evaluation metrics (MASE, sMAPE, MSIS, etc.)
- Uncertainty quantification for forecasts
- Proper multivariate time series handling
- Backward compatible with all your current functions
"""