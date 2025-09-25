# -*- coding: utf-8 -*-
"""GMA sampling test 9: Hierarchical Bayesian linear regression [new, used].ipynb

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

improvements ON 16/08/2025: \\

(1) pre-compute densities \\
(2) add the correct 1+ to gradient \\
(3) use projected GD \\
(4) use decaying learning rate \\
(5) use MC gradient estimator \\
(6) add bechmarks: ADVI (MFVI, PyMC) + GM-ADVI. \\

## tuning GMA variance.
"""

# Set random seed for reproducibility
np.random.seed(111)
torch.manual_seed(111)
rng_key = jax.random.PRNGKey(111)
rng = np.random.RandomState(111)

# --- GMA Hyperparameters ---
N = 1500  # Total number of Gaussian components
M = 100   # Samples per Gaussian
K = 1500  # Number of iterations for weight updates
eta = 0.1  # Initial learning rate
cov_scale = 0.0001 # Use a small covariance for high resolution

# --- Initialize fixed means and covariances ---
initial_means = np.random.multivariate_normal(np.zeros(d), 5 * np.eye(d), size=N)
initial_covariances = [cov_scale * np.eye(d) for _ in range(N)]
initial_weights = np.full(N, 1/N)

# --- Initialize samples from each Gaussian component ---
samples = np.zeros((N, M, d))
for i in range(N):
    samples[i] = np.random.multivariate_normal(mean=initial_means[i], cov=initial_covariances[i], size=M)
flat_samples = samples.reshape(N * M, d)

# --- Run GMA ---
# CORRECTED projection
def project_to_simplex(v):
    """
    Correct and robust implementation of the projection onto the probability simplex.
    """
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0

    # The original algorithm correctly finds rho without the need for a special check.
    # If cond is all False, this will correctly result in an index error,
    # but for a valid probability simplex projection, rho will always be found.
    # In practice with floating point numbers, the algorithm is stable.
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

start_time_GMA = time.time()

# 1. Pre-compute GMM PDFs
pdf_matrix_P = np.zeros((N * M, N))
for l in tqdm(range(N), desc="GMA Pre-computing PDFs"):
    pdf_matrix_P[:, l] = multivariate_normal.pdf(flat_samples, mean=initial_means[l], cov=initial_covariances[l])

# 2. Pre-compute log target densities (using training data)
log_p_target = log_unnormalized_p(flat_samples, log_radon_train, floor_measure_train, county_train, u_county_level)

# Initialize weights
weights = np.zeros((N, K+1))
weights[:, 0] = initial_weights

# 3. Iteratively update weights
for k in tqdm(range(1, K+1), desc="GMA Iterations (pGD)"):
    q_values = pdf_matrix_P @ weights[:, k-1]
    gradient = np.zeros(N)
    for i in range(N):
        start_idx, end_idx = i * M, (i + 1) * M
        log_q_slice = np.log(q_values[start_idx:end_idx] + 1e-9)
        log_p_slice = log_p_target[start_idx:end_idx]
        gradient[i] = 1 + (1/M) * np.sum(log_q_slice - log_p_slice)

    if np.isnan(gradient).any():
        print(f"Warning: NaN gradient at iteration {k}. Skipping update.")
        weights[:, k] = weights[:, k-1]
        continue

    learning_rate = eta / k
    intermediate_weights = weights[:, k-1] - learning_rate * gradient
    weights[:, k] = project_to_simplex(intermediate_weights)

end_time_GMA = time.time()
gma_time = end_time_GMA - start_time_GMA
print(f"GMA approximation time: {gma_time:.4f} seconds")

# --- Ensemble sampling ---
total_samples = N*M
final_weights = weights[:, -1]
selected_indices = rng.choice(N, total_samples, p=final_weights, replace=True)
sample_indices = np.random.randint(0, M, size=total_samples)
gma_samples = samples[selected_indices, sample_indices]
np.save('gma_samples.npy', gma_samples)

# Plot GMA weight evolution
for i in range(N):
    plt.plot(range(K+1), weights[i, :])
plt.xlabel('Iteration')
plt.ylabel('Weight value')
plt.title('Evolution of Component Weights')
plt.grid(True)
plt.tight_layout()
plt.show()


# ##################################################################
# ## EVALUATION AND VISUALIZATION ##
# ##################################################################
# Store all execution times
execution_times = {
    'GMA': gma_time, 'MH': mh_time, 'HMC': hmc_time, 'MFVI-ADVI': advi_time
}
np.save('execution_times.npy', execution_times)


all_samples = {
    'GMA': gma_samples,
    'MH': mh_samples,
    'HMC (NUTS)': hmc_samples,
    'MFVI-ADVI': advi_samples
}

# --- Posterior Predictive Check on Test Set ---
print("\n--- Posterior Predictive Check on Test Set ---")
predictive_results = {}
for method, samples_data in all_samples.items():
    # Thin samples for large sample sets to make prediction faster
    sample_subset = samples_data[::10] if len(samples_data) > 10000 else samples_data

    gamma0_s = sample_subset[:, 0]
    gamma1_s = sample_subset[:, 1]
    beta_s = sample_subset[:, 2]
    sigma_a_s = sample_subset[:, 3]
    epsilon_a_s = sample_subset[:, 5:]

    mu_a_s = gamma0_s[:, np.newaxis] + gamma1_s[:, np.newaxis] * u_county_level
    alpha_county_s = mu_a_s + sigma_a_s[:, np.newaxis] * epsilon_a_s
    alpha_obs_s = alpha_county_s[:, county_test]
    y_hat_test = alpha_obs_s + beta_s[:, np.newaxis] * floor_measure_test
    y_pred_mean = y_hat_test.mean(axis=0)

    rmse = np.sqrt(mean_squared_error(log_radon_test, y_pred_mean))
    predictive_results[method] = {'preds': y_pred_mean, 'rmse': rmse}
    print(f"{method} Test RMSE: {rmse:.4f}")

"""## optimal."""

import numpy as np
from scipy.stats import multivariate_normal, norm, halfcauchy, uniform
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import torch
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jax_norm
import blackjax
import pymc as pm
import optax
import copy
import warnings
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
import seaborn as sns
import pandas as pd
from sklearn.model_selection import train_test_split
import copy

# Suppress verbose output from PyMC
warnings.filterwarnings("ignore", category=FutureWarning)

# Set random seed for reproducibility
np.random.seed(111)
torch.manual_seed(111)
rng_key = jax.random.PRNGKey(111)
# Use a simple integer seed for PyMC compatibility
RANDOM_SEED = 111
rng = np.random.RandomState(RANDOM_SEED)


# ##################################################################
# ## DATA PREPARATION: RADON DATASET ##
# ##################################################################
print("--- Loading and Preprocessing Radon Data ---")
path = "https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data/srrs2.dat"
radon_df = pd.read_csv(path)
radon_df.columns = radon_df.columns.map(str.strip)
srrs_mn = radon_df[radon_df.state == "MN"].copy()

cty_path = "https://raw.githubusercontent.com/pymc-devs/pymc-examples/main/examples/data/cty.dat"
cty = pd.read_csv(cty_path)
cty.columns = cty.columns.map(str.strip)

srrs_mn["fips"] = srrs_mn.stfips * 1000 + srrs_mn.cntyfips
cty_mn = cty[cty.st == "MN"].copy()
cty_mn["fips"] = 1000 * cty_mn.stfips + cty_mn.ctfips
srrs_mn = srrs_mn.merge(cty_mn[["fips", "Uppm"]], on="fips")
srrs_mn = srrs_mn.drop_duplicates(subset="idnum")
srrs_mn.county = srrs_mn.county.map(str.strip)

# Create county lookup
county_names = srrs_mn.county.unique()
county_lookup = {name: i for i, name in enumerate(county_names)}
srrs_mn["county_code"] = srrs_mn.county.replace(county_lookup).values

# Prepare data for model
srrs_mn["log_radon"] = np.log(srrs_mn.activity + 0.1)
srrs_mn["log_uranium"] = np.log(srrs_mn.Uppm)

# Create a county-level uranium vector
u_county_level = srrs_mn.groupby('county_code')['Uppm'].mean().apply(np.log).values

n_counties = len(county_names)
d = 5 + n_counties # 5 global params (gamma0, gamma1, beta, sigma_a, sigma_y) + 85 county effects

# --- Train-Test Split ---
train_df, test_df = train_test_split(srrs_mn, test_size=0.2, random_state=RANDOM_SEED)

log_radon_train = train_df.log_radon.values
floor_measure_train = train_df.floor.values
county_train = train_df.county_code.values

log_radon_test = test_df.log_radon.values
floor_measure_test = test_df.floor.values
county_test = test_df.county_code.values

print(f"Data loaded. Train size: {len(train_df)}, Test size: {len(test_df)}, Counties: {n_counties}")
print(f"Total parameters to infer: {d}")


# ##################################################################
# ## EXPLORATORY DATA ANALYSIS (EDA) ##
# ##################################################################
print("\n--- Exploratory Data Analysis of Log-Radon Levels ---")
# 1. Calculate and print the variation of the response variable
# We do this on the full dataset before splitting to understand the overall problem.
log_radon_all = srrs_mn["log_radon"].values
mean_log_radon = np.mean(log_radon_all)
std_dev_log_radon = np.std(log_radon_all)
print(f"Mean log-radon level: {mean_log_radon:.4f}")
print(f"Standard Deviation of log-radon: {std_dev_log_radon:.4f}")
print("This standard deviation is a key baseline. A good model should have a final RMSE lower than this value.")
print("It represents the error you would get if you simply predicted the mean value for every household.")
# 2. Visualize the distribution of the response variable
plt.figure(figsize=(10, 6))
sns.histplot(log_radon_all, kde=True, bins=30, label='Distribution of log-radon')
# Add vertical lines for the mean and one standard deviation
plt.axvline(mean_log_radon, color='r', linestyle='-', linewidth=2, label=f'Mean ({mean_log_radon:.2f})')
plt.axvline(mean_log_radon + std_dev_log_radon, color='g', linestyle='--', linewidth=2, label=f'±1 Std Dev ({std_dev_log_radon:.2f})')
plt.axvline(mean_log_radon - std_dev_log_radon, color='g', linestyle='--')
# Final plot adjustments
plt.title('Distribution of Log-Radon Levels in Minnesota Households', fontsize=16)
plt.xlabel('Log-Radon Measurement', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.legend()
plt.grid(axis='y', alpha=0.5)
plt.show()

# ##################################################################
# ## HIERARCHICAL BAYESIAN LINEAR REGRESSION POSTERIOR ##
# ##################################################################

# --- Custom JAX HalfCauchy logpdf ---
def jax_halfcauchy_logpdf(x, scale=1.0):
    return jnp.where(x >= 0, jnp.log(2) - jnp.log(jnp.pi) - jnp.log(scale) - jnp.log(1 + (x/scale)**2), -jnp.inf)

# --- Custom PyTorch HalfCauchy logpdf ---
def torch_halfcauchy_logpdf(x, scale=1.0):
    return torch.where(x >= 0, np.log(2) - np.log(np.pi) - np.log(scale) - torch.log(1 + (x/scale)**2), -torch.inf)


# --- Target distribution functions for different libraries ---

def log_unnormalized_p(params, y_data, x_data, county_idx, u_county):
    # NumPy version for MH, GMA
    params = np.atleast_2d(params)

    # Unpack parameters
    gamma0, gamma1, beta, sigma_a, sigma_y = params[:, 0], params[:, 1], params[:, 2], params[:, 3], params[:, 4]
    epsilon_a = params[:, 5:]

    # Add constraint checks for standard deviation parameters.
    valid_mask = (sigma_a > 0) & (sigma_y > 0) & (sigma_y < 100)
    log_p = np.full(params.shape[0], -1e9)

    if not np.any(valid_mask):
        return log_p

    p_valid = params[valid_mask]
    gamma0, gamma1, beta, sigma_a, sigma_y = p_valid[:, 0], p_valid[:, 1], p_valid[:, 2], p_valid[:, 3], p_valid[:, 4]
    epsilon_a = p_valid[:, 5:]

    # Log Priors
    log_p_gamma0 = norm.logpdf(gamma0, 0, 10)
    log_p_gamma1 = norm.logpdf(gamma1, 0, 10)
    log_p_beta = norm.logpdf(beta, 0, 10)
    log_p_sigma_a = halfcauchy.logpdf(sigma_a, scale=5)
    log_p_sigma_y = uniform.logpdf(sigma_y, loc=0, scale=100)
    log_p_epsilon_a = np.sum(norm.logpdf(epsilon_a, 0, 1), axis=1)

    log_prior = log_p_gamma0 + log_p_gamma1 + log_p_beta + log_p_sigma_a + log_p_sigma_y + log_p_epsilon_a

    # Log Likelihood
    mu_a = gamma0[:, np.newaxis] + gamma1[:, np.newaxis] * u_county
    alpha_county = mu_a + sigma_a[:, np.newaxis] * epsilon_a
    alpha_obs = alpha_county[:, county_idx]

    y_hat = alpha_obs + beta[:, np.newaxis] * x_data
    log_likelihood = np.sum(norm.logpdf(y_data, loc=y_hat, scale=sigma_y[:, np.newaxis]), axis=1)

    log_p[valid_mask] = log_prior + log_likelihood
    return log_p

def log_unnormalized_p_torch(params):
    # PyTorch version for LMC
    gamma0, gamma1, beta, sigma_a, sigma_y = params[:5]
    epsilon_a = params[5:]

    # --- DEBUGGED SECTION: Add constraint checks for PyTorch ---
    if not (sigma_a > 0 and sigma_y > 0 and sigma_y < 100):
        # Return a tensor that is still connected to the graph to avoid grad_fn error
        return torch.tensor(-torch.inf) + 0 * torch.sum(params)
    # --- END DEBUGGED SECTION ---

    # Log Priors
    log_p_gamma0 = torch.distributions.Normal(0, 10).log_prob(gamma0)
    log_p_gamma1 = torch.distributions.Normal(0, 10).log_prob(gamma1)
    log_p_beta = torch.distributions.Normal(0, 10).log_prob(beta)
    log_p_sigma_a = torch_halfcauchy_logpdf(sigma_a, scale=5)
    log_p_sigma_y = torch.where((sigma_y > 0) & (sigma_y < 100), -torch.log(torch.tensor(100.0)), -torch.inf)
    log_p_epsilon_a = torch.sum(torch.distributions.Normal(0, 1).log_prob(epsilon_a))

    log_prior = log_p_gamma0 + log_p_gamma1 + log_p_beta + log_p_sigma_a + log_p_sigma_y + log_p_epsilon_a

    # Log Likelihood (using training data)
    u_county_level_torch = torch.tensor(u_county_level, dtype=torch.float32)
    county_train_torch = torch.tensor(county_train, dtype=torch.long)
    floor_measure_train_torch = torch.tensor(floor_measure_train, dtype=torch.float32)
    log_radon_train_torch = torch.tensor(log_radon_train, dtype=torch.float32)

    mu_a = gamma0 + gamma1 * u_county_level_torch
    alpha_county = mu_a + sigma_a * epsilon_a
    alpha_obs = alpha_county[county_train_torch]

    y_hat = alpha_obs + beta * floor_measure_train_torch
    log_likelihood = torch.sum(torch.distributions.Normal(y_hat, sigma_y).log_prob(log_radon_train_torch))

    return log_prior + log_likelihood

# ##################################################################
# ## GMA SAMPLING (pGD with True Projection) ##
# ##################################################################

# --- GMA Hyperparameters ---
N = 1500  # Total number of Gaussian components
M = 100   # Samples per Gaussian
K = 1500  # Number of iterations for weight updates
eta = 0.1  # Initial learning rate
cov_scale = 0.0001 # Use a small covariance for high resolution

# --- Initialize fixed means and covariances ---
initial_means = np.random.multivariate_normal(np.zeros(d), 5 * np.eye(d), size=N)
initial_covariances = [cov_scale * np.eye(d) for _ in range(N)]
initial_weights = np.full(N, 1/N)

# --- Initialize samples from each Gaussian component ---
samples = np.zeros((N, M, d))
for i in range(N):
    samples[i] = np.random.multivariate_normal(mean=initial_means[i], cov=initial_covariances[i], size=M)
flat_samples = samples.reshape(N * M, d)

# (CORRECTED) simplex projection
def project_to_simplex(v):
    """
    Correct and robust implementation of the projection onto the probability simplex.
    """
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0

    # The original algorithm correctly finds rho without the need for a special check.
    # If cond is all False, this will correctly result in an index error,
    # but for a valid probability simplex projection, rho will always be found.
    # In practice with floating point numbers, the algorithm is stable.
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

start_time_GMA = time.time()

# 1. Pre-compute GMM PDFs
pdf_matrix_P = np.zeros((N * M, N))
for l in tqdm(range(N), desc="GMA Pre-computing PDFs"):
    pdf_matrix_P[:, l] = multivariate_normal.pdf(flat_samples, mean=initial_means[l], cov=initial_covariances[l])

# 2. Pre-compute log target densities (using training data)
log_p_target = log_unnormalized_p(flat_samples, log_radon_train, floor_measure_train, county_train, u_county_level)

# Initialize weights
weights = np.zeros((N, K+1))
weights[:, 0] = initial_weights

# 3. Iteratively update weights
for k in tqdm(range(1, K+1), desc="GMA Iterations (pGD)"):
    q_values = pdf_matrix_P @ weights[:, k-1]
    gradient = np.zeros(N)
    for i in range(N):
        start_idx, end_idx = i * M, (i + 1) * M
        log_q_slice = np.log(q_values[start_idx:end_idx] + 1e-9)
        log_p_slice = log_p_target[start_idx:end_idx]
        gradient[i] = 1 + (1/M) * np.sum(log_q_slice - log_p_slice)

    if np.isnan(gradient).any():
        print(f"Warning: NaN gradient at iteration {k}. Skipping update.")
        weights[:, k] = weights[:, k-1]
        continue

    learning_rate = eta / k
    intermediate_weights = weights[:, k-1] - learning_rate * gradient
    weights[:, k] = project_to_simplex(intermediate_weights)

end_time_GMA = time.time()
gma_time = end_time_GMA - start_time_GMA
print(f"GMA approximation time: {gma_time:.4f} seconds")

# --- Ensemble sampling ---
total_samples = N*M
final_weights = weights[:, -1]
selected_indices = rng.choice(N, total_samples, p=final_weights, replace=True)
sample_indices = np.random.randint(0, M, size=total_samples)
gma_samples = samples[selected_indices, sample_indices]
np.save('gma_samples.npy', gma_samples)

# Plot GMA weight evolution
for i in range(N):
    plt.plot(range(K+1), weights[i, :])
plt.xlabel('Iteration')
plt.ylabel('Weight value')
plt.title('Evolution of Component Weights')
plt.grid(True)
plt.tight_layout()
plt.show()

# ##################################################################
# ## BENCHMARKING ALGORITHMS ##
# ##################################################################

### Metropolis-Hastings ###
def mh_sampler(n_iter, initial_point=np.zeros(d), proposal_cov=np.eye(d)):
    accepted = []
    current = np.array(initial_point)
    current_log_p = log_unnormalized_p(current, log_radon_train, floor_measure_train, county_train, u_county_level)
    for _ in tqdm(range(n_iter), desc="MH Sampling"):
        proposal = current + np.random.multivariate_normal(np.zeros(d), proposal_cov)
        proposal_log_p = log_unnormalized_p(proposal, log_radon_train, floor_measure_train, county_train, u_county_level)
        ratio = np.exp(proposal_log_p - current_log_p)
        if np.random.rand() < min(1.0, ratio):
            current = proposal
            current_log_p = proposal_log_p
        accepted.append(current)
    return np.array(accepted)

print("\n--- Running Metropolis-Hastings ---")
start_time_mh = time.time()
mh_samples = mh_sampler(n_iter=total_samples, initial_point=np.zeros(d), proposal_cov=0.01*np.eye(d))
end_time_mh = time.time()
mh_time = end_time_mh - start_time_mh
print(f"MH time: {mh_time:.4f} seconds")
np.save('mh_samples.npy', mh_samples)


### HMC: PyMC Implementations ###
coords = {"county": county_names, "obs_id": np.arange(len(log_radon_train))}
with pm.Model(coords=coords) as hierarchical_model:
    county_idx_data = pm.Data("county_idx_data", county_train, dims="obs_id")
    floor_measure_data = pm.Data("floor_measure_data", floor_measure_train, dims="obs_id")

    sigma_a = pm.HalfCauchy("sigma_a", 5)
    gamma_0 = pm.Normal("gamma_0", mu=0.0, sigma=10.0)
    gamma_1 = pm.Normal("gamma_1", mu=0.0, sigma=10.0)
    mu_a = pm.Deterministic("mu_a", gamma_0 + gamma_1 * u_county_level, dims="county")
    epsilon_a = pm.Normal("epsilon_a", mu=0, sigma=1, dims="county")
    alpha = pm.Deterministic("alpha", mu_a + sigma_a * epsilon_a, dims="county")
    beta = pm.Normal("beta", mu=0.0, sigma=10.0)
    sigma_y = pm.Uniform("sigma_y", lower=0, upper=100)
    y_hat = alpha[county_idx_data] + beta * floor_measure_data
    y_like = pm.Normal("y_like", mu=y_hat, sigma=sigma_y, observed=log_radon_train, dims="obs_id")

# --- HMC (NUTS) ---
print("\n--- Running HMC (NUTS) with PyMC ---")
start_time_hmc = time.time()
with hierarchical_model:
    hmc_trace = pm.sample(total_samples, tune=2000, chains=1, random_seed=RANDOM_SEED)
end_time_hmc = time.time()
hmc_time = end_time_hmc - start_time_hmc
print(f"HMC time: {hmc_time:.4f} seconds")
hmc_samples = np.vstack([
    hmc_trace.posterior["gamma_0"].values.flatten(),
    hmc_trace.posterior["gamma_1"].values.flatten(),
    hmc_trace.posterior["beta"].values.flatten(),
    hmc_trace.posterior["sigma_a"].values.flatten(),
    hmc_trace.posterior["sigma_y"].values.flatten(),
    hmc_trace.posterior["epsilon_a"].values.reshape(total_samples, -1).T
]).T
np.save('hmc_samples.npy', hmc_samples)


# --- MFVI-ADVI ---
print("\n--- Running MFVI-ADVI with PyMC ---")
start_time_advi = time.time()
with hierarchical_model:
    mean_field_approx = pm.fit(n=50000, method='advi', random_seed=RANDOM_SEED)
    advi_trace = mean_field_approx.sample(total_samples)
end_time_advi = time.time()
advi_time = end_time_advi - start_time_advi
print(f"ADVI time: {advi_time:.4f} seconds")
advi_samples = np.vstack([
    advi_trace.posterior["gamma_0"].values.flatten(),
    advi_trace.posterior["gamma_1"].values.flatten(),
    advi_trace.posterior["beta"].values.flatten(),
    advi_trace.posterior["sigma_a"].values.flatten(),
    advi_trace.posterior["sigma_y"].values.flatten(),
    advi_trace.posterior["epsilon_a"].values.reshape(total_samples, -1).T
]).T
np.save('advi_samples.npy', advi_samples)


# ##################################################################
# ## EVALUATION AND VISUALIZATION ##
# ##################################################################
# Store all execution times
execution_times = {
    'GMA': gma_time, 'MH': mh_time, 'HMC': hmc_time, 'MFVI-ADVI': advi_time
}
np.save('execution_times.npy', execution_times)


all_samples = {
    'GMA': gma_samples,
    'MH': mh_samples,
    'HMC (NUTS)': hmc_samples,
    'MFVI-ADVI': advi_samples
}

# --- 1. Posterior Predictive Check on Test Set ---
print("\n--- Posterior Predictive Check on Test Set ---")
predictive_results = {}
for method, samples_data in all_samples.items():
    # Thin samples for large sample sets to make prediction faster
    sample_subset = samples_data[::10] if len(samples_data) > 10000 else samples_data

    gamma0_s = sample_subset[:, 0]
    gamma1_s = sample_subset[:, 1]
    beta_s = sample_subset[:, 2]
    sigma_a_s = sample_subset[:, 3]
    epsilon_a_s = sample_subset[:, 5:]

    mu_a_s = gamma0_s[:, np.newaxis] + gamma1_s[:, np.newaxis] * u_county_level
    alpha_county_s = mu_a_s + sigma_a_s[:, np.newaxis] * epsilon_a_s
    alpha_obs_s = alpha_county_s[:, county_test]
    y_hat_test = alpha_obs_s + beta_s[:, np.newaxis] * floor_measure_test
    y_pred_mean = y_hat_test.mean(axis=0)

    rmse = np.sqrt(mean_squared_error(log_radon_test, y_pred_mean))
    predictive_results[method] = {'preds': y_pred_mean, 'rmse': rmse}
    print(f"{method} Test RMSE: {rmse:.4f}")

# --- 2. Plotting Key Parameter Posteriors ---
param_indices = {
    r'$\gamma_0$': 0,
    r'$\gamma_1$': 1,
    r'$\beta$': 2,
    r'$\sigma_{\alpha}$': 3,
    r'$\sigma_y$': 4
    # We will now create the county offset plot separately
}
param_names = list(param_indices.keys())

# --- Add a name for our derived quantity ---
derived_param_name = r'County Offset 1 ($\sigma_{\alpha} \cdot \epsilon_1$)'
param_names.append(derived_param_name)


fig, axes = plt.subplots(len(param_names), 1, figsize=(8, 15), sharex=False)
for i, name in enumerate(param_names):
    ax = axes[i]

    # --- Check if we are plotting a base parameter or the derived one ---
    if name in param_indices:
        # --- Plot the prior distribution for base parameters ---
        if name in [r'$\gamma_0$', r'$\gamma_1$', r'$\beta$']:
            x = np.linspace(-30, 30, 200)
            p = norm.pdf(x, 0, 10)
            ax.plot(x, p, color='gray', linestyle='--', linewidth=2.5, label='Prior')
        elif name == r'$\sigma_{\alpha}$':
            x = np.linspace(0, 25, 200)
            p = halfcauchy.pdf(x, scale=5)
            ax.plot(x, p, color='gray', linestyle='--', linewidth=2.5, label='Prior')
        elif name == r'$\sigma_y$':
            x = np.linspace(-10, 110, 200)
            p = uniform.pdf(x, loc=0, scale=100)
            ax.plot(x, p, color='gray', linestyle='--', linewidth=2.5, label='Prior')

        # Plot the posteriors for each method from the stored samples
        for method, samples_data in all_samples.items():
            sns.kdeplot(samples_data[:, param_indices[name]], ax=ax, label=method, fill=True, alpha=0.35)

    elif name == derived_param_name:
        # --- MODIFIED: Plot the posterior of the DERIVED county offset ---
        # We don't plot a single prior line for derived quantities.
        ax.axvline(0, color='gray', linestyle='--', linewidth=2.5, label='Prior Mean (0)')

        for method, samples_data in all_samples.items():
            # Derive the posterior samples for the unscaled offset
            # sigma_alpha is at index 3, epsilon_1 is at index 5
            unscaled_offset_samples = samples_data[:, 3] * samples_data[:, 5]

            sns.kdeplot(unscaled_offset_samples, ax=ax, label=method, fill=True, alpha=0.35)

    ax.set_title(f'Prior and Posterior of {name}')
    ax.legend()
plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

# --- 3. Plot County-Level Effects ---
fig, ax = plt.subplots(figsize=(10, 8))
ax.set_title('County Intercepts vs. Uranium Level')

# HMC (gold standard)
hmc_gamma_0_mean = hmc_samples[:, param_indices[r'$\gamma_0$']].mean()
hmc_gamma_1_mean = hmc_samples[:, param_indices[r'$\gamma_1$']].mean()
hmc_sigma_a_mean = hmc_samples[:, param_indices[r'$\sigma_{\alpha}$']].mean()
hmc_epsilon_a_mean = hmc_samples[:, 5:].mean(axis=0)
hmc_alphas = hmc_gamma_0_mean + hmc_gamma_1_mean * u_county_level + hmc_sigma_a_mean * hmc_epsilon_a_mean
ax.plot(u_county_level, hmc_gamma_0_mean + hmc_gamma_1_mean * u_county_level, 'k--', label='HMC Mean Trend')
ax.scatter(u_county_level, hmc_alphas, label='HMC County Intercepts', alpha=0.8, marker='o')

# GMA
gma_gamma_0_mean = gma_samples[:, param_indices[r'$\gamma_0$']].mean()
gma_gamma_1_mean = gma_samples[:, param_indices[r'$\gamma_1$']].mean()
gma_sigma_a_mean = gma_samples[:, param_indices[r'$\sigma_{\alpha}$']].mean()
gma_epsilon_a_mean = gma_samples[:, 5:].mean(axis=0)
gma_alphas = gma_gamma_0_mean + gma_gamma_1_mean * u_county_level + gma_sigma_a_mean * gma_epsilon_a_mean
ax.scatter(u_county_level, gma_alphas, label='GMA County Intercepts', alpha=0.6, marker='x')

# MH
mh_gamma_0_mean = mh_samples[:, param_indices[r'$\gamma_0$']].mean()
mh_gamma_1_mean = mh_samples[:, param_indices[r'$\gamma_1$']].mean()
mh_sigma_a_mean = mh_samples[:, param_indices[r'$\sigma_{\alpha}$']].mean()
mh_epsilon_a_mean = mh_samples[:, 5:].mean(axis=0)
mh_alphas = mh_gamma_0_mean + mh_gamma_1_mean * u_county_level + mh_sigma_a_mean * mh_epsilon_a_mean
ax.scatter(u_county_level, mh_alphas, label='MH County Intercepts', alpha=0.6, marker='d')

# MFVI-ADVI
advi_gamma_0_mean = advi_samples[:, param_indices[r'$\gamma_0$']].mean()
advi_gamma_1_mean = advi_samples[:, param_indices[r'$\gamma_1$']].mean()
advi_sigma_a_mean = advi_samples[:, param_indices[r'$\sigma_{\alpha}$']].mean()
advi_epsilon_a_mean = advi_samples[:, 5:].mean(axis=0)
advi_alphas = advi_gamma_0_mean + advi_gamma_1_mean * u_county_level + advi_sigma_a_mean * advi_epsilon_a_mean
ax.scatter(u_county_level, advi_alphas, label='MFVI-ADVI County Intercepts', alpha=0.6, marker='v')

ax.set_xlabel('County-level log-uranium')
ax.set_ylabel('Estimated County Intercept (alpha_j)')
ax.legend()
plt.show()

# --- 4. Plot Posterior Predictive Check ---
fig, ax = plt.subplots(figsize=(8, 8))
min_val = min(log_radon_test.min(), min(p['preds'].min() for p in predictive_results.values()))
max_val = max(log_radon_test.max(), max(p['preds'].max() for p in predictive_results.values()))
ax.plot([min_val, max_val], [min_val, max_val], 'k--', label='Ideal (y=x)')

for method, results in predictive_results.items():
    ax.scatter(log_radon_test, results['preds'], alpha=0.5, label=f"{method} (RMSE: {results['rmse']:.3f})")

ax.set_xlabel("True Log-Radon (Test Set)")
ax.set_ylabel("Predicted Log-Radon (Posterior Mean)")
ax.set_title("Posterior Predictive Check on Held-Out Data")
ax.legend()
ax.set_aspect('equal', adjustable='box')
plt.show()

"""## posterior plot for paper."""

# --- Create the plot with the desired 2x3 grid structure ---
fig, axes = plt.subplots(2, 3, figsize=(18, 10))
axes = axes.flatten()  # Flatten the 2x3 grid into a 1D array

for i, name in enumerate(param_names):
    ax = axes[i]

    # --- NEW: Create an adaptive x-range based on the posterior data ---
    # First, collect all posterior samples for the current parameter to find their range.
    if name == derived_param_name:
        posterior_data = np.concatenate([
            samples_data[:, 3] * samples_data[:, 5] for samples_data in all_samples.values()
        ])
    else:
        posterior_data = np.concatenate([
            samples_data[:, param_indices[name]] for samples_data in all_samples.values()
        ])

    # Determine the plotting range with 15% padding on each side
    data_min, data_max = posterior_data.min(), posterior_data.max()
    padding = (data_max - data_min) * 0.15
    x_range = np.linspace(data_min - padding, data_max + padding, 300)
    # --- END NEW ---

    # --- Plot using the new adaptive range 'x_range' ---
    if name in param_indices:
        # Plot the prior distribution for base parameters over the adaptive range
        if name in [r'$\gamma_0$', r'$\gamma_1$', r'$\beta$']:
            p = norm.pdf(x_range, 0, 10)
            ax.plot(x_range, p, color='gray', linestyle='--', linewidth=2.5, label='Prior')
        elif name == r'$\sigma_{\alpha}$':
            # Ensure we only plot the HalfCauchy for x >= 0
            x_pos = x_range[x_range >= 0]
            p = halfcauchy.pdf(x_pos, scale=5)
            ax.plot(x_pos, p, color='gray', linestyle='--', linewidth=2.5, label='Prior')
        elif name == r'$\sigma_y$':
            p = uniform.pdf(x_range, loc=0, scale=100)
            ax.plot(x_range, p, color='gray', linestyle='--', linewidth=2.5, label='Prior')

        # Plot the posteriors for each method
        for method, samples_data in all_samples.items():
            sns.kdeplot(samples_data[:, param_indices[name]], ax=ax, label=method, fill=True, alpha=0.35)

    elif name == derived_param_name:
        # Plot the posterior of the DERIVED county offset
        ax.axvline(0, color='gray', linestyle='--', linewidth=2.5, label='Prior Mean (0)')

        for method, samples_data in all_samples.items():
            unscaled_offset_samples = samples_data[:, 3] * samples_data[:, 5]
            sns.kdeplot(unscaled_offset_samples, ax=ax, label=method, fill=True, alpha=0.35)

    ax.set_title(f'Prior and Posterior of {name}', fontsize=12)
    ax.legend()
    # Enforce the calculated x-limits for consistency
    ax.set_xlim(x_range[0], x_range[-1])

plt.tight_layout(rect=[0, 0.03, 1, 0.95])
plt.show()

"""# download results to local."""

!zip -r /content/colab_files.zip /content
from google.colab import files
files.download('/content/colab_files.zip')

"""# end."""
