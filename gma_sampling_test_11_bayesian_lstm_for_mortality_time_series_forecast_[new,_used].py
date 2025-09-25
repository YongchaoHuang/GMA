# -*- coding: utf-8 -*-
"""GMA sampling test 11: Bayesian LSTM for mortality time series forecast [new, used].ipynb

#yongchao.huang@abdn.ac.uk
"""

!lscpu

import psutil

print("CPU cores:", psutil.cpu_count(logical=False))
print("Logical CPUs:", psutil.cpu_count(logical=True))
print("Memory (GB):", round(psutil.virtual_memory().total / 1e9, 2))
print("Disk space (GB):", round(psutil.disk_usage('/').total / 1e9, 2))

"""# begin."""

pip install blackjax

"""# Final version.

Improvements ON 16/08/2025: \\

(1) pre-compute densities \\

(2) add the correct 1+ to gradient \\

(3) use projected GD \\

(4) use decaying learning rate \\

(5) use MC gradient estimator \\

(6) add bechmarks: ADVI (MFVI, PyMC) + GM-ADVI. \\
"""

import psutil, os, gc
print("RAM GB:", round(psutil.virtual_memory().total/1e9,2))
# later, to free:
gc.collect()

import jax
from jax import config
config.update("jax_enable_x64", False)  # stay in float32

"""## Data preparation, pre-processing and EDA."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import warnings
import time
from tqdm import tqdm

import torch
import torch.nn as nn
import pymc as pm
import pytensor.tensor as pt
import pytensor
from scipy.stats import multivariate_normal, norm, halfnorm
from scipy.special import logsumexp
import blackjax
from sklearn.metrics import mean_squared_error

# --- Setup and Configuration ---
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=UserWarning)

# ###################################################################
# ## EXPERIMENT HYPERPARAMETERS ##
# ###################################################################

# (This section is unchanged, but included for context)
np.random.seed(111)
torch.manual_seed(111)
RANDOM_SEED = 111
rng = np.random.RandomState(RANDOM_SEED)

COUNTRY = 'United States'
LOOK_BACK = 52       # Use previous 52 weeks (1 year) to predict the next
N_FORECAST = 52      # Forecast 52 weeks (1 year) into the future
HIDDEN_DIM = 16
N_LAYERS = 3
TOTAL_SAMPLES = 3000
N_TUNE = 1000

# ###################################################################
# ## 1. DATA ACQUISITION AND PREPROCESSING ##
# ###################################################################
print(f"--- 1. Loading and Preprocessing WEEKLY Mortality Data for {COUNTRY} ---")

try:
    file_path = 'world_mortality.csv'
    df_raw = pd.read_csv(file_path)

    # --- Step 1: Filter for weekly data for the target country ---
    country_df = df_raw[(df_raw['country_name'] == COUNTRY) & (df_raw['time_unit'] == 'weekly')].copy()

    # --- Step 2: Create a proper datetime index ---
    country_df['date'] = pd.to_datetime(country_df['year'].astype(str) + '-' + country_df['time'].astype(str) + '-0', format='%Y-%W-%w')

    # --- Step 3: Aggregate duplicate weeks by summing their death counts ---
    weekly_deaths = country_df.groupby('date')['deaths'].sum().reset_index()
    weekly_deaths = weekly_deaths.sort_values('date').set_index('date')

    # --- Step 4: Ensure complete weekly frequency and interpolate gaps ---
    weekly_idx = pd.date_range(weekly_deaths.index.min(), weekly_deaths.index.max(), freq='W-SUN')
    country_df = weekly_deaths.reindex(weekly_idx)
    country_df['deaths'].interpolate(method='linear', inplace=True)

    # --- Step 5: Construct Mortality Index ---
    deaths = country_df['deaths'].values
    mean_deaths = np.mean(deaths)
    mortality_index = deaths / mean_deaths
    log_mortality_index = np.log(mortality_index)

    processed_df = pd.DataFrame({
        'deaths': deaths,
        'mortality_index': mortality_index,
        'log_mortality_index': log_mortality_index
    }, index=country_df.index)
    print(f"Data prepared for {COUNTRY} from {processed_df.index.min().date()} to {processed_df.index.max().date()} ({len(processed_df)} weeks).")

except Exception as e:
    print(f"\nFATAL ERROR: Could not process the weekly data. Halting script. Error: {e}")
    raise e

# --- Step 6: Create Sequences and Train/Test Split ---
def create_sequences(data, look_back):
    X, y = [], []
    for i in range(len(data) - look_back):
        X.append(data[i:(i + look_back)])
        y.append(data[i + look_back])
    return np.array(X), np.array(y)
X, y = create_sequences(log_mortality_index, LOOK_BACK)

if len(X) <= N_FORECAST:
    raise ValueError(f"Not enough data to create a train/test split. Have only {len(X)} sequences.")

test_size = N_FORECAST
X_train, X_test = X[:-test_size], X[-test_size:]
y_train, y_test = y[:-test_size], y[-test_size:]
y_test_unscaled_index = np.exp(y_test)

# --- Step 7: Standardize Data ---
feature_scaler = StandardScaler()
target_scaler = StandardScaler()

# Reshape the training features into a single column to learn one mean/std
X_train_reshaped = X_train.reshape(-1, 1)
feature_scaler.fit(X_train_reshaped)

# Transform the train and test sets (after reshaping) and then reshape them back to sequences
X_train_scaled = feature_scaler.transform(X_train_reshaped).reshape(X_train.shape)
X_test_scaled = feature_scaler.transform(X_test.reshape(-1, 1)).reshape(X_test.shape)

# Fit and transform the targets
y_train_scaled = target_scaler.fit_transform(y_train.reshape(-1, 1)).flatten()
print(f"Split data into {len(X_train)} training and {len(X_test)} testing sequences.")


# ###################################################################
# ## 2. EXPLORATORY DATA ANALYSIS (EDA) ##
# ###################################################################
print("\n--- 2. Exploratory Data Analysis ---")

# Create visualizations for the new weekly time series
fig, axes = plt.subplots(1, 3, figsize=(21, 5))

# 1. Weekly death counts
axes[0].plot(processed_df.index, processed_df['deaths'], color='darkred')
axes[0].set_title('Weekly Death Counts', fontsize=12, fontweight='bold')
axes[0].set_xlabel('Date')
axes[0].set_ylabel('Total Deaths')
axes[0].grid(True, alpha=0.3)

# 2. Normalized Weekly Mortality Index
axes[1].plot(processed_df.index, processed_df['mortality_index'], color='darkblue')
axes[1].axhline(1.0, color='gray', linestyle='--', label='Long-term Average')
axes[1].set_title('Normalized Weekly Mortality Index', fontsize=12, fontweight='bold')
axes[1].set_xlabel('Date')
axes[1].set_ylabel('Index (Deaths / Mean)')
axes[1].legend()
axes[1].grid(True, alpha=0.3)

# 3. Log-Mortality Index (the series the model will see)
axes[2].plot(processed_df.index, processed_df['log_mortality_index'], color='darkgreen')
axes[2].set_title('Log-Mortality Index (Model Input)', fontsize=12, fontweight='bold')
axes[2].set_xlabel('Date')
axes[2].set_ylabel('Log(Index)')
axes[2].grid(True, alpha=0.3)

fig.suptitle(f"EDA for Weekly Mortality Data in {COUNTRY}", fontsize=16)
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()






### genetic visualisation func for later comparison use ###
def plot_forecast_comparison(
    processed_df,
    y_train,
    train_preds,
    y_test,
    test_preds,
    method_name,
    classic_lstm_preds=None
):
    """
    Generates a comprehensive time series plot comparing a model's fit and forecast
    against the ground truth, while also calculating and reporting performance metrics.
    """
    # --- NEW: Calculate RMSE internally ---
    # RMSE for the in-sample training fit
    train_rmse = np.sqrt(mean_squared_error(y_train, train_preds))

    # Handle both Bayesian (2D array) and point forecasts (1D array) for test RMSE
    if test_preds.ndim == 2: # Bayesian samples with uncertainty
        mean_forecast = test_preds.mean(axis=0)
        test_rmse = np.sqrt(mean_squared_error(y_test, mean_forecast))
    else: # Point forecast
        mean_forecast = test_preds
        test_rmse = np.sqrt(mean_squared_error(y_test, mean_forecast))

    print("\n" + "-"*50)
    print(f"--- Performance Metrics for: {method_name} ---")
    print(f"Train Set RMSE: {train_rmse:.4f}")
    print(f"Test Set RMSE:  {test_rmse:.4f}")
    print("-"*50)
    # --- END NEW ---

    plt.figure(figsize=(15, 7))

    # --- Plot Ground Truth ---
    plt.plot(processed_df.index, processed_df['mortality_index'],
             label='Ground Truth', color='black', alpha=0.8)

    # --- Plot In-Sample Fit ---
    train_dates = processed_df.index[LOOK_BACK:len(y_train) + LOOK_BACK]
    plt.plot(train_dates, train_preds,
             label=f'{method_name} Train Fit', color='blue', linestyle='--')

    # --- Plot Out-of-Sample Forecast ---
    test_dates = processed_df.index[-len(y_test):]
    plt.plot(test_dates, mean_forecast,
             label=f'{method_name} Mean Forecast', color='red', linestyle='--')

    # Plot credible interval only if Bayesian samples are provided
    if test_preds.ndim == 2:
        ci_lower = np.percentile(test_preds, 5, axis=0)
        ci_upper = np.percentile(test_preds, 95, axis=0)
        plt.fill_between(test_dates, ci_lower, ci_upper,
                         color='red', alpha=0.2, label='90% Credible Interval')

    # --- Overlay Classic LSTM Benchmark if provided ---
    if classic_lstm_preds is not None:
        plt.plot(test_dates, classic_lstm_preds,
                 label='Classic LSTM Forecast', color='purple', linestyle=':')

    # --- Formatting ---
    split_date = train_dates[-1]
    plt.axvline(x=split_date, color='gray', linestyle=':',
                linewidth=2, label='Train/Test Split')

    # Use the internally calculated test_rmse for the title
    title = f'Model Performance: {method_name} (Test RMSE: {test_rmse:.4f})'

    plt.title(title, fontsize=16, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Mortality Index')
    plt.legend()
    plt.grid(True, alpha=0.5)
    plt.show()

"""## Classic LSTM.

"""

# ##################################################################
# ## BENCHMARK: CLASSIC LSTM WITH BACKPROPAGATION  (flexible forecasts)
# ##################################################################
print("\n--- Training a Classic LSTM Benchmark ---")

# --- Set random seeds for reproducible model initialization ---
np.random.seed(111)
torch.manual_seed(111)
RANDOM_SEED = 111

# Choose forecasting modes for classic LSTM as well
TRAIN_FORECAST_MODE = "rolling"   # "rolling" or "one_step"
TEST_FORECAST_MODE  = "rolling"   # "rolling" or "one_step"

class LSTMModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, num_layers, output_dim):
        super(LSTMModel, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        # x: (B, T, 1)
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_dim)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(out[:, -1, :])  # last step
        return out  # (B, 1)

pytorch_model = LSTMModel(input_dim=1, hidden_dim=HIDDEN_DIM, output_dim=1, num_layers=N_LAYERS)

def get_param_dim(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def set_params(model, params_vec):
    offset = 0
    for p in model.parameters():
        if p.requires_grad:
            numel = p.numel()
            p.data.copy_(torch.tensor(params_vec[offset:offset+numel]).view(p.shape))
            offset += numel

param_dim = get_param_dim(pytorch_model)
d = param_dim + 1
print(f"LSTM model has {param_dim} parameters. Total inferred dimension: {d}")

classic_lstm_model = LSTMModel(input_dim=1, hidden_dim=HIDDEN_DIM, output_dim=1, num_layers=N_LAYERS)

# -------------------
# Train classic LSTM
# -------------------
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(classic_lstm_model.parameters(), lr=0.01)
num_epochs = 1000

# Prepare data for PyTorch training
X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1)  # (N, T, 1)
y_train_torch = torch.tensor(y_train_scaled, dtype=torch.float32).unsqueeze(-1)  # (N, 1)

loss_history = []
start_time_classic = time.time()
for epoch in range(num_epochs):
    classic_lstm_model.train()
    outputs = classic_lstm_model(X_train_torch)
    loss = criterion(outputs, y_train_torch)
    loss_history.append(loss.item())
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {loss.item():.6f}')
classic_lstm_time = time.time() - start_time_classic
print(f"Classic LSTM training time: {classic_lstm_time:.4f} seconds")

# -------------------------------------------------------
# Forecast helpers (support rolling vs one-step)
# -------------------------------------------------------
@torch.no_grad()
def _model_last_step_mu_classic(model, window_scaled_1d):
    """Given a scaled sliding window (shape [LOOK_BACK]), return next-step mean (scaled)."""
    x = torch.tensor(window_scaled_1d, dtype=torch.float32).view(1, -1, 1)
    out = model(x)                    # (1,1)
    return float(out.item())          # scaled log prediction

def _scaled_to_log_classic(pred_scaled):
    """Inverse-transform a SINGLE scaled target to log-space scalar."""
    return float(target_scaler.inverse_transform(np.array(pred_scaled, ndmin=2))[:, 0])

def forecast_train_classic(model, mode: str):
    model.eval()
    if mode == "one_step":
        X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1)
        preds_scaled = model(X_train_torch).detach().numpy().reshape(-1)   # <-- added .detach()
        preds_log = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        return preds_log
    elif mode == "rolling":
        window = X_train_scaled[0].copy()
        out = []
        for _ in range(len(y_train)):
            mu_scaled = _model_last_step_mu_classic(model, window)
            mu_log = _scaled_to_log_classic([[mu_scaled]])
            out.append(mu_log)
            window = np.roll(window, -1)
            window[-1] = mu_scaled
        return np.array(out, dtype=np.float32)
    else:
        raise ValueError("mode must be 'one_step' or 'rolling'")

def forecast_test_classic(model, mode: str):
    model.eval()
    if mode == "one_step":
        X_test_torch = torch.tensor(X_test_scaled, dtype=torch.float32).unsqueeze(-1)
        preds_scaled = model(X_test_torch).detach().numpy().reshape(-1)   # <-- added .detach()
        preds_log = target_scaler.inverse_transform(preds_scaled.reshape(-1, 1)).flatten()
        return preds_log
    elif mode == "rolling":
        window = X_train_scaled[-1].copy()
        out = []
        for _ in range(N_FORECAST):
            mu_scaled = _model_last_step_mu_classic(model, window)
            mu_log = _scaled_to_log_classic([[mu_scaled]])
            out.append(mu_log)
            window = np.roll(window, -1)
            window[-1] = mu_scaled
        return np.array(out, dtype=np.float32)
    else:
        raise ValueError("mode must be 'one_step' or 'rolling'")

# ---------------------------------------------
# Generate train/test predictions (LOG & INDEX)
# ---------------------------------------------
print("Generating in-sample (training) predictions with "
      f"{TRAIN_FORECAST_MODE} mode...")
train_preds_log = forecast_train_classic(classic_lstm_model, TRAIN_FORECAST_MODE)
train_preds_index = np.exp(train_preds_log)

print("Generating out-of-sample (test) predictions with "
      f"{TEST_FORECAST_MODE} mode...")
test_preds_log = forecast_test_classic(classic_lstm_model, TEST_FORECAST_MODE)
classic_forecast_index = np.exp(test_preds_log)

# ---------------------------------------------
# Visualize & RMSE (reuse your plot function)
# ---------------------------------------------
print("\n--- Visualizing Classic LSTM Fit and Forecast ---")

# 1) Training loss curve
plt.figure(figsize=(10, 5))
plt.plot(range(num_epochs), loss_history, label='Training Loss')
plt.title('Classic LSTM Training Loss Over Epochs', fontsize=14, fontweight='bold')
plt.xlabel('Epoch'); plt.ylabel('MSE Loss'); plt.legend(); plt.grid(True, alpha=0.5)
plt.show()

# 2) Unified plot via your helper
plot_forecast_comparison(
    processed_df=processed_df,
    y_train=np.exp(y_train),
    train_preds=train_preds_index,
    y_test=y_test_unscaled_index,
    test_preds=classic_forecast_index,
    method_name=f"Classic LSTM ({TRAIN_FORECAST_MODE} train, {TEST_FORECAST_MODE} test)"
)

"""## GMA."""

# =========================================================
# === Bayesian LSTM with GMA (MD or pGD) ==================
# === Train/Test: rolling vs one-step forecasting =========
# =========================================================
import time, gc
import numpy as np
import torch
from tqdm import tqdm
from scipy.stats import norm, halfnorm
from scipy.special import logsumexp
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# ==============================
# User-selectable FLEX knobs
# ==============================
OPT_METHOD = "MD"          # "MD" (mirror descent) or "PGD" (projected GD)
TRAIN_FORECAST_MODE = "rolling"   # "rolling" or "one_step"
TEST_FORECAST_MODE  = "rolling"    # "rolling" or "one_step"

# =========================================================
# Helpers: params, posterior, projections, predictions
# =========================================================
def flatten_params(model) -> np.ndarray:
    """Flatten PyTorch model params (requires_grad=True) into 1D numpy."""
    with torch.no_grad():
        return np.concatenate([p.detach().cpu().numpy().ravel()
                               for p in model.parameters() if p.requires_grad])

def log_unnormalized_p(params):
    """
    Unnormalized log-posterior: log p(theta, sigma | data)
    Prior: theta ~ N(0, 1), sigma ~ HalfNormal(1)
    Likelihood: y | theta, sigma ~ N(mu_theta(X), sigma) on TRAINING set (scaled y).
    """
    theta, log_sigma = params[:-1], params[-1]
    sigma = np.exp(log_sigma)
    if sigma <= 0 or not np.isfinite(sigma):
        return -np.inf

    set_params(pytorch_model, theta)

    # Priors
    log_prior_theta = norm.logpdf(theta, 0.0, 1.0).sum()
    log_prior_sigma = halfnorm.logpdf(sigma, scale=1.0)

    # Likelihood on training data (scaled)
    X_torch = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1)
    with torch.no_grad():
        mu_scaled = pytorch_model(X_torch).cpu().numpy().reshape(-1)
        # robust: some LSTMs return (n,1); others (n,T,1). We take last step if needed:
        if mu_scaled.size != y_train_scaled.size:
            mu_scaled = pytorch_model(X_torch).reshape(len(X_train_scaled), -1).cpu().numpy()[:, -1]
    log_lik = norm.logpdf(y_train_scaled, loc=mu_scaled, scale=sigma).sum()

    return log_prior_theta + log_prior_sigma + log_lik

def project_to_simplex_correct(v):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    ind = np.arange(n_features) + 1
    cond = u - (cssv - 1) / ind > 0
    rho = ind[cond][-1]
    theta = (cssv[rho - 1] - 1) / rho
    projected = np.maximum(v - theta, 0)
    return projected

def _model_last_step_mu(model, window_scaled_1d):
    """
    Given a 1D sliding window (scaled), produce the next-step mean (scaled).
    Works whether model returns (1,1), (1,T,1), (1,T) ... we take the last scalar.
    """
    x = torch.tensor(window_scaled_1d, dtype=torch.float32).view(1, -1, 1)
    with torch.no_grad():
        out = model(x)
        out = out.reshape(1, -1)[:, -1]  # last step
        mu = float(out.item())
    return mu

def _scaled_to_log(pred_scaled):
    """Invert target scaler to log-space scalar."""
    return float(target_scaler.inverse_transform(np.array(pred_scaled, ndmin=2))[:, 0])

def _forecast_single_draw(model, theta_plus, n_forecast, mode, add_noise=False, seed_window_scaled=None, X_one_step=None):
    """
    Produce a 1D array of LOG-forecasts for one posterior draw.
    - mode == "rolling": use 'seed_window_scaled' (shape [LOOK_BACK]) and roll forward autoregressively.
    - mode == "one_step": use 'X_one_step' (shape [n, LOOK_BACK]) — teacher forcing per step.
    theta_plus includes ...[:-1] = theta, last element = log_sigma (for optional noise if desired).
    """
    theta, log_sigma = theta_plus[:-1], theta_plus[-1]
    sigma = float(np.exp(log_sigma))
    set_params(model, theta)

    if mode == "rolling":
        assert seed_window_scaled is not None and seed_window_scaled.shape[0] == LOOK_BACK
        window = seed_window_scaled.copy()
        out_log = []
        for _ in range(n_forecast):
            mu_scaled = _model_last_step_mu(model, window)
            if add_noise:
                mu_scaled_noisy = mu_scaled + np.random.normal(0.0, sigma)
            else:
                mu_scaled_noisy = mu_scaled
            mu_log = _scaled_to_log([[mu_scaled_noisy]])
            out_log.append(mu_log)
            window = np.roll(window, -1)
            window[-1] = mu_scaled
        return np.array(out_log, dtype=np.float32)

    elif mode == "one_step":
        assert X_one_step is not None and X_one_step.ndim == 2 and X_one_step.shape[1] == LOOK_BACK
        out_log = []
        for i in range(X_one_step.shape[0]):
            window = X_one_step[i]
            mu_scaled = _model_last_step_mu(model, window)
            mu_scaled_noisy = mu_scaled + np.random.normal(0.0, sigma) if add_noise else mu_scaled
            mu_log = _scaled_to_log([[mu_scaled_noisy]])
            out_log.append(mu_log)
        return np.array(out_log, dtype=np.float32)

    else:
        raise ValueError("mode must be 'rolling' or 'one_step'.")

def posterior_forecasts(
    model, samples, train_mode, test_mode,
    n_forecast_test,
    seed_window_train=None, seed_window_test=None,
    X_train_one_step=None, X_test_one_step=None,
    max_draws=500, rng=np.random.default_rng(123)
):
    """
    Produce posterior forecast matrices (S x T_len) in LOG space for train & test.
    """
    S = min(max_draws, samples.shape[0])
    idx = rng.choice(samples.shape[0], S, replace=False)

    T_train = len(y_train) if train_mode == "rolling" else X_train_one_step.shape[0]
    T_test  = n_forecast_test if test_mode == "rolling" else X_test_one_step.shape[0]

    train_mat = np.empty((S, T_train), dtype=np.float32)
    test_mat  = np.empty((S, T_test),  dtype=np.float32)

    for si, s in enumerate(idx):
        theta_plus = samples[s]

        if train_mode == "rolling":
            train_mat[si] = _forecast_single_draw(
                model, theta_plus, n_forecast=T_train, mode="rolling", add_noise=False,
                seed_window_scaled=seed_window_train
            )
        else:
            train_mat[si] = _forecast_single_draw(
                model, theta_plus, n_forecast=T_train, mode="one_step", add_noise=False,
                X_one_step=X_train_one_step
            )

        if test_mode == "rolling":
            test_mat[si] = _forecast_single_draw(
                model, theta_plus, n_forecast=T_test, mode="rolling", add_noise=False,
                seed_window_scaled=seed_window_test
            )
        else:
            test_mat[si] = _forecast_single_draw(
                model, theta_plus, n_forecast=T_test, mode="one_step", add_noise=False,
                X_one_step=X_test_one_step
            )

    return train_mat, test_mat  # both in LOG space

# =========================================================
# 1) Warm start from Classic LSTM (MAP-like θ*)
# =========================================================
theta_map = flatten_params(classic_lstm_model)

# --- RECOMPUTE training predictions here so this block is self-contained ---
X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1)
with torch.no_grad():
    train_preds_scaled = classic_lstm_model(X_train_torch).detach().cpu().numpy().ravel()

# Rough sigma from training MSE (on the scaled target)
train_mse = float(np.mean((train_preds_scaled - y_train_scaled.ravel())**2))
sigma_hat = np.sqrt(max(train_mse, 1e-8))
log_sigma_map = np.log(sigma_hat)

# Augment with log-sigma
theta_map_plus = np.concatenate([theta_map, np.array([log_sigma_map])], axis=0).astype(np.float32)
d = int(theta_map_plus.size)  # θ plus logσ

# =========================================================
# 2) GMA Hyperparameters
# =========================================================
N, M, K = 200, 30, 100           # (smaller for speed while prototyping)
eta0, k0 = 0.05, 800                # MD step size base (MD will use eta0/sqrt(k+k0))
sigma2_const = np.float32(5e-4)     # local cloud variance (precomputable)

beta_min = 0.30
tau0     = 1.8
lam0     = 1e-2
alpha0   = 0.20
L_avg    = 75

rng = np.random.default_rng(111)

# =========================================================
# 3) Initialize component means around θ* (small jitter)
# =========================================================
init_scale_theta  = 0.02
init_scale_logsig = 0.05

initial_means = np.empty((N, d), dtype=np.float32)
for i in range(N):
    jitter        = rng.normal(0.0, init_scale_theta,  size=d-1).astype(np.float32)
    logsig_jitter = rng.normal(0.0, init_scale_logsig, size=1    ).astype(np.float32)
    mean_i   = theta_map_plus.copy()
    mean_i[:-1] += jitter
    mean_i[-1]  += logsig_jitter
    initial_means[i] = mean_i

# =========================================================
# 4) Draw M local samples per component (σ² I)
# =========================================================
flat_samples = np.empty((N * M, d), dtype=np.float32)
for i in range(N):
    s, t = i * M, (i + 1) * M
    flat_samples[s:t] = rng.normal(loc=initial_means[i],
                                   scale=np.sqrt(sigma2_const),
                                   size=(M, d)).astype(np.float32)

# =========================================================
# 5) Target log-density for all samples
# =========================================================
print("\nEvaluating target log-density for all local samples...")
log_p_target = np.array([log_unnormalized_p(s) for s in flat_samples], dtype=np.float32)
if not np.all(np.isfinite(log_p_target)):
    print("WARNING: non-finite log_p_target; clipping.")
    log_p_target = np.clip(log_p_target, -1e30, 1e30).astype(np.float32)

# Standardize once
mu_lp = log_p_target.mean()
sd_lp = log_p_target.std() + 1e-8

# =========================================================
# 6) Precompute log N(x; μ_l, σ^2 I) for all x, components
# =========================================================
print("Precomputing Gaussian log-PDF matrix...")
cst     = (-0.5 * (d * np.log(2.0 * np.pi * sigma2_const))).astype(np.float32)
x_norm2 = np.einsum('ij,ij->i', flat_samples, flat_samples, optimize=True).astype(np.float32)
mu_norm2= np.einsum('ij,ij->i', initial_means, initial_means, optimize=True).astype(np.float32)
xdotmu  = (flat_samples @ initial_means.T).astype(np.float32)  # (NM, N)

log_pdf_matrix_L = (cst - 0.5 * (x_norm2[:, None] + mu_norm2[None, :]) / sigma2_const
                    + (xdotmu / sigma2_const)).astype(np.float32)
if not np.all(np.isfinite(log_pdf_matrix_L)):
    print("WARNING: non-finite log_pdf_matrix_L; clipping.")
    log_pdf_matrix_L = np.clip(log_pdf_matrix_L, -1e30, 1e30).astype(np.float32)

del xdotmu
gc.collect()

# =========================================================
# 7) Optimizer over mixture weights: MD or pGD
# =========================================================
print(f"\n--- Running GMA ({OPT_METHOD}) — warm start + anti-collapse schedules ---")
weights   = np.full((N, K + 1), 1.0 / N, dtype=np.float32)
ent_hist  = np.empty(K, dtype=np.float32)
effk_hist = np.empty(K, dtype=np.float32)

start_time_GMA = time.time()
for k in tqdm(range(1, K + 1), desc=f"GMA {OPT_METHOD} iterations"):

    # Common schedules
    frac   = k / K
    beta   = beta_min + (1.0 - beta_min) * frac
    tau_k  = 1.0 + (tau0 - 1.0) * (1.0 - frac)   # used by MD
    lam_k  = lam0 * (1.0 - frac)
    alpha_k= max(0.05, alpha0 * (1.0 - frac))    # used by MD

    eta_k = np.float32(eta0 / np.sqrt(k + k0))

    w_prev = weights[:, k - 1]
    log_w_prev = np.log(np.maximum(w_prev, 1e-12)).astype(np.float32)

    # log q(x) for all samples
    log_q_values = logsumexp(log_w_prev[None, :] + log_pdf_matrix_L, axis=1).astype(np.float32)
    if not np.all(np.isfinite(log_q_values)):
        log_q_values = np.clip(log_q_values, -1e30, 1e30).astype(np.float32)

    # Monte Carlo gradient (tempered + standardized)
    diffs = (log_q_values - (beta * log_p_target - mu_lp) / sd_lp).astype(np.float32)

    grad = np.empty(N, dtype=np.float32)
    for i in range(N):
        s, t = i * M, (i + 1) * M
        grad[i] = 1.0 + (1.0 / M) * np.sum(diffs[s:t], dtype=np.float32)

    # + entropy gradient contribution
    grad += lam_k * (1.0 + log_w_prev)

    if OPT_METHOD == "MD":
        # Mirror step + temperature + convex mixing
        log_w_tilde = (log_w_prev - eta_k * grad) / tau_k
        w_prop = np.exp(log_w_tilde - logsumexp(log_w_tilde)).astype(np.float32)
        w_next = (1 - alpha_k) * w_prev + alpha_k * w_prop
        w_next = np.maximum(w_next, 0).astype(np.float32)
        w_next /= w_next.sum()

    else:  # PGD: Euclidean step then simplex projection
        w_prop = w_prev - eta_k * grad
        w_prop = project_to_simplex_correct(w_prop)
        w_next = w_prop

    weights[:, k] = w_next

    # diagnostics
    H = -(w_next * np.log(np.maximum(w_next, 1e-12))).sum(dtype=np.float32)
    ent_hist[k - 1]  = H
    effk_hist[k - 1] = np.exp(H)
    if k % 50 == 0:
        if OPT_METHOD == "MD":
            print(f"[k={k:4d}] H(w)={H:.3f}, eff#comp≈{effk_hist[k-1]:.1f}, "
                  f"beta={beta:.2f}, tau={tau_k:.2f}, lam={lam_k:.4f}, alpha={alpha_k:.3f}, eta={eta_k:.5f}")
        else:
            print(f"[k={k:4d}] H(w)={H:.3f}, eff#comp≈{effk_hist[k-1]:.1f}, "
                  f"beta={beta:.2f}, lam={lam_k:.4f}, eta={eta_k:.5f}")

gma_time = time.time() - start_time_GMA
print(f"GMA ({OPT_METHOD}) time: {gma_time:.2f}s")

# =========================================================
# 8) Tail-averaged final weights & posterior sampling
# =========================================================
L = min(L_avg, K)
final_weights = weights[:, -L:].mean(axis=1).astype(np.float32)
final_weights = np.maximum(final_weights, 0)
final_weights /= final_weights.sum()

top10_idx = np.argsort(-final_weights)[:10]
print("\nTop-10 mixture components by final (tail-averaged) weight:")
for r, idx in enumerate(top10_idx, 1):
    print(f"{r:2d}. Component {idx} — weight {final_weights[idx]:.6f}")

selected_components = rng.choice(N, size=TOTAL_SAMPLES, p=final_weights, replace=True)
rows = selected_components * M + rng.integers(0, M, size=TOTAL_SAMPLES)
gma_samples = flat_samples[rows].astype(np.float32)

# save
np.save('weights.npy', weights.astype(np.float32))
np.save('gma_samples.npy', gma_samples.astype(np.float32))

# =========================================================
# 9) Diagnostics plots
# =========================================================
plt.figure(figsize=(10,4))
plt.plot(ent_hist, label='Entropy H(w^k)')
plt.plot(effk_hist, label='Effective #components exp(H)', linestyle='--')
plt.xlabel('Iteration k'); plt.grid(True, alpha=0.4); plt.legend()
plt.title(f'GMA {OPT_METHOD} Diagnostics'); plt.tight_layout(); plt.show()

plt.figure(figsize=(8,4))
for i in range(N):
    plt.plot(weights[i, :], alpha=0.25)
plt.xlabel('Iteration'); plt.ylabel('Weight')
plt.title(f'GMA ({OPT_METHOD}) weight evolution'); plt.tight_layout(); plt.show()

# =========================================================
# 10) Unified train/test predictive summaries (LOG→INDEX)
# =========================================================
print("\n--- Forecasting with posterior samples ---")

seed_window_train = X_train_scaled[0]  if TRAIN_FORECAST_MODE == "rolling" else None
seed_window_test  = X_train_scaled[-1] if TEST_FORECAST_MODE  == "rolling" else None

X_train_one_step = X_train_scaled if TRAIN_FORECAST_MODE == "one_step" else None
X_test_one_step  = X_test_scaled  if TEST_FORECAST_MODE  == "one_step" else None

train_forecasts_log, test_forecasts_log = posterior_forecasts(
    model=pytorch_model,
    samples=gma_samples,
    train_mode=TRAIN_FORECAST_MODE,
    test_mode=TEST_FORECAST_MODE,
    n_forecast_test=N_FORECAST,
    seed_window_train=seed_window_train,
    seed_window_test=seed_window_test,
    X_train_one_step=X_train_one_step,
    X_test_one_step=X_test_one_step,
    max_draws=500
)

train_forecasts_index = np.exp(train_forecasts_log)
test_forecasts_index  = np.exp(test_forecasts_log)

train_mean_index = train_forecasts_index.mean(axis=0)
train_ci_lo = np.percentile(train_forecasts_index, 5, axis=0)
train_ci_hi = np.percentile(train_forecasts_index, 95, axis=0)

test_mean_index = test_forecasts_index.mean(axis=0)
test_ci_lo = np.percentile(test_forecasts_index, 5, axis=0)
test_ci_hi = np.percentile(test_forecasts_index, 95, axis=0)

theta_mean = gma_samples.mean(axis=0)[:-1]
set_params(pytorch_model, theta_mean)
with torch.no_grad():
    preds_scaled = pytorch_model(torch.tensor(X_train_scaled, dtype=torch.float32).unsqueeze(-1))
preds_log = target_scaler.inverse_transform(preds_scaled.numpy()).flatten()
posterior_mean_fit_index = np.exp(preds_log)

train_dates = processed_df.index[LOOK_BACK:LOOK_BACK + len(y_train)]
test_dates  = processed_df.index[-len(y_test_unscaled_index):]
split_date  = train_dates[-1]

rmse_train = np.sqrt(mean_squared_error(np.exp(y_train)[:len(train_mean_index)], train_mean_index))
rmse_test  = np.sqrt(mean_squared_error(np.exp(y_test)[:len(test_mean_index)],   test_mean_index))
print(f"RMSE (mean path) — Train: {rmse_train:.4f} | Test: {rmse_test:.4f}")

plt.figure(figsize=(16, 7))
plt.plot(processed_df.index, processed_df['mortality_index'],
         color='black', linewidth=1.8, alpha=0.85, label='Ground Truth')

plt.plot(train_dates[:len(train_mean_index)], train_mean_index, color='tab:blue', linestyle='--',
         linewidth=2.0, label=f'GMA Train Predictive Mean ({TRAIN_FORECAST_MODE})')
plt.fill_between(train_dates[:len(train_mean_index)], train_ci_lo, train_ci_hi, color='tab:blue',
                 alpha=0.15, label='GMA Train 90% CI')

plt.plot(train_dates[:len(posterior_mean_fit_index)], posterior_mean_fit_index, color='tab:blue', linestyle=':',
         linewidth=1.8, alpha=0.9, label='Posterior-mean Fit')

plt.plot(test_dates[:len(test_mean_index)], test_mean_index, color='tab:red', linestyle='--',
         linewidth=2.0, label=f'GMA Test Predictive Mean ({TEST_FORECAST_MODE})')
plt.fill_between(test_dates[:len(test_mean_index)], test_ci_lo, test_ci_hi, color='tab:red', alpha=0.18,
                 label='GMA Test 90% CI')

plt.axvline(split_date, color='gray', linestyle=':', linewidth=2, label='Train/Test Split')
plt.title(f'GMA {OPT_METHOD}: Train/Test Predictive', fontsize=15, fontweight='bold')
plt.xlabel('Date'); plt.ylabel('Mortality Index')
plt.grid(True, alpha=0.4)
plt.legend(ncol=2)
plt.tight_layout()
plt.show()

"""# top 10 weights plot."""

import matplotlib.pyplot as plt

# Data
components = [127, 69, 57, 82, 87, 138, 118, 88, 12, 28]
weights = [0.422310, 0.016253, 0.014613, 0.013915, 0.013547, 0.013507, 0.013184, 0.013137, 0.012990, 0.012789]

# Bar chart
plt.figure(figsize=(10, 5))
plt.bar([str(c) for c in components], weights)
plt.xlabel('Component Index')
plt.ylabel('Final Weight')
plt.title('Top-10 GMA Mixture Components by Final (Tail-Averaged) Weight')
plt.tight_layout()

"""# download results to local."""

!zip -r /content/colab_files.zip /content
from google.colab import files
files.download('/content/colab_files.zip')

"""# end."""
