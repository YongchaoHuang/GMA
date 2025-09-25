# -*- coding: utf-8 -*-
"""GMA sampling test 8: BLR [new, used].ipynb

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

## tuning variance.
"""

# Set random seed for reproducibility
np.random.seed(111)
torch.manual_seed(111)
rng_key = jax.random.PRNGKey(111)
rng = np.random.RandomState(111)

# --- GMA Hyperparameters ---
N = 2000  # Total number of Gaussian components
M = 50   # Samples per Gaussian
K = 1500  # Number of iterations for weight updates
eta = 0.1  # Initial learning rate
cov_scale = 0.03 # Use a small covariance for high resolution

# --- Initialize fixed means and covariances ---
# BLR CHANGE: Initialize means from a broad Gaussian in 4D
initial_means = np.random.multivariate_normal(np.zeros(d), 5 * np.eye(d), size=N)
initial_covariances = [cov_scale * np.eye(d) for _ in range(N)]
initial_weights = np.full(N, 1/N)

# --- Initialize samples from each Gaussian component ---
samples = np.zeros((N, M, d))
for i in range(N):
    samples[i] = np.random.multivariate_normal(mean=initial_means[i], cov=initial_covariances[i], size=M)
flat_samples = samples.reshape(N * M, d)

# --- Run GMA ---
def project_to_simplex(v):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

start_time_GMA = time.time()

# 1. Pre-compute GMM PDFs
pdf_matrix_P = np.zeros((N * M, N))
for l in tqdm(range(N), desc="GMA Pre-computing PDFs"):
    pdf_matrix_P[:, l] = multivariate_normal.pdf(flat_samples, mean=initial_means[l], cov=initial_covariances[l])

# 2. Pre-compute log target densities
log_p_target = log_unnormalized_p(flat_samples, X_train_scaled, y_train)

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
if not np.isclose(np.sum(final_weights), 1.0):
    raise ValueError(f"final_weights: Expected sum to be 1.0, got {np.sum(final_weights)}")
# save weights and gma_samples
np.save('weights.npy', weights)
np.save('gma_samples.npy', gma_samples)


# --- Plot GMA Details ---
# Calculate posterior mean and mode for the GMA samples
gma_w_mean = np.mean(gma_samples, axis=0)
gma_w_mode = np.zeros(d)
for i in range(d):
    counts, bin_edges = np.histogram(gma_samples[:, i], bins=50)
    max_bin_index = np.argmax(counts)
    gma_w_mode[i] = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2

# Create a single figure for the two side-by-side joint plots
fig = plt.figure(figsize=(18, 8))
fig.suptitle(f'GMA Sampling Results (N={N}, K={K}, M={M}, Cov={cov_scale}, eta={eta}, time={gma_time:.2f}s)', fontsize=16)

# Define axis limits based on GMA samples for consistent scaling within this plot
padding_w01 = (gma_samples[:, 0].max() - gma_samples[:, 0].min()) * 0.1
padding_w23 = (gma_samples[:, 2].max() - gma_samples[:, 2].min()) * 0.1
xlim01 = (gma_samples[:, 0].min() - padding_w01, gma_samples[:, 0].max() + padding_w01)
ylim01 = (gma_samples[:, 1].min() - padding_w01, gma_samples[:, 1].max() + padding_w01)
xlim23 = (gma_samples[:, 2].min() - padding_w23, gma_samples[:, 2].max() + padding_w23)
ylim23 = (gma_samples[:, 3].min() - padding_w23, gma_samples[:, 3].max() + padding_w23)

# Create Plot for (w_0, w_1) on the left
gs1 = fig.add_gridspec(2, 2, left=0.05, right=0.48, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
ax1 = fig.add_subplot(gs1[1, 0])
ax1_histx = fig.add_subplot(gs1[0, 0], sharex=ax1)
ax1_histy = fig.add_subplot(gs1[1, 1], sharey=ax1)
ax1_histx.tick_params(axis="x", labelbottom=False)
ax1_histy.tick_params(axis="y", labelleft=False)

ax1.scatter(flat_samples[:, 0], flat_samples[:, 1], alpha=0.05, color='gray', s=5, label='Initial Samples')
ax1.scatter(gma_samples[:, 0], gma_samples[:, 1], alpha=0.2, s=10, label='Ensemble Samples')
ax1.scatter(gma_w_mean[0], gma_w_mean[1], marker='o', s=150, color='orange', zorder=10, label='Posterior Mean')
ax1.scatter(gma_w_mode[0], gma_w_mode[1], marker='X', s=150, color='orange', zorder=10, label='Posterior Mode')
ax1.scatter(mle_coefs[0], mle_coefs[1], marker='P', s=200, color='red', zorder=10, label='MLE')
ax1.set_xlabel('$w_0$')
ax1.set_ylabel('$w_1$')
ax1.legend()
ax1.set_xlim(xlim01)
ax1.set_ylim(ylim01)
ax1_histx.hist(gma_samples[:, 0], bins=30, density=True)
ax1_histy.hist(gma_samples[:, 1], bins=30, orientation='horizontal', density=True)

# Create Plot for (w_2, w_3) on the right
gs2 = fig.add_gridspec(2, 2, left=0.55, right=0.98, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
ax2 = fig.add_subplot(gs2[1, 0])
ax2_histx = fig.add_subplot(gs2[0, 0], sharex=ax2)
ax2_histy = fig.add_subplot(gs2[1, 1], sharey=ax2)
ax2_histx.tick_params(axis="x", labelbottom=False)
ax2_histy.tick_params(axis="y", labelleft=False)

ax2.scatter(flat_samples[:, 2], flat_samples[:, 3], alpha=0.05, color='gray', s=5, label='Initial Samples')
ax2.scatter(gma_samples[:, 2], gma_samples[:, 3], alpha=0.2, s=10, label='Ensemble Samples')
ax2.scatter(gma_w_mean[2], gma_w_mean[3], marker='o', s=150, color='orange', zorder=10, label='Posterior Mean')
ax2.scatter(gma_w_mode[2], gma_w_mode[3], marker='X', s=150, color='orange', zorder=10, label='Posterior Mode')
ax2.scatter(mle_coefs[2], mle_coefs[3], marker='P', s=200, color='red', zorder=10, label='MLE')
ax2.set_xlabel('$w_2$')
ax2.set_ylabel('$w_3$')
ax2.legend()
ax2.set_xlim(xlim23)
ax2.set_ylim(ylim23)
ax2_histx.hist(gma_samples[:, 2], bins=30, density=True)
ax2_histy.hist(gma_samples[:, 3], bins=30, orientation='horizontal', density=True)

plt.show()


# --- Create a separate figure for the weight evolution ---
fig_weights, ax_weights = plt.subplots(1, 1, figsize=(8, 6))
for i in range(N):
    ax_weights.plot(range(K+1), weights[i, :])
ax_weights.set_xlabel('Iteration')
ax_weights.set_ylabel('Weight value')
ax_weights.set_title('Evolution of Component Weights')
ax_weights.grid(True)
plt.tight_layout()
plt.show()

"""## optimal."""

import numpy as np
from scipy.stats import multivariate_normal, norm, gaussian_kde
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
from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import seaborn as sns
import pandas as pd
from sklearn.linear_model import LogisticRegression

# Suppress verbose output from PyMC
warnings.filterwarnings("ignore", category=FutureWarning)

# Set random seed for reproducibility
np.random.seed(111)
torch.manual_seed(111)
rng_key = jax.random.PRNGKey(111)
rng = np.random.RandomState(111)

# ##################################################################
# ## DATA PREPARATION: IRIS DATASET ##
# ##################################################################
# Load Iris dataset
iris = load_iris()
X, y = iris.data, iris.target

# Use only two classes (Setosa vs. Versicolor) to make it a binary problem
X = X[y != 2]
y = y[y != 2]
d = X.shape[1] # Dimension is now 4

# Split and scale the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=111)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# ##################################################################
# ## BAYESIAN LOGISTIC REGRESSION POSTERIOR ##
# ##################################################################

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def jax_sigmoid(x):
    return 1 / (1 + jnp.exp(-x))

def torch_sigmoid(x):
    return 1 / (1 + torch.exp(-x))

# --- Target distribution functions for different libraries ---

def log_unnormalized_p(w, X_data, y_data):
    # NumPy version for MH, GMA
    w = np.atleast_2d(w)
    # Log prior: Standard normal N(0, I)
    log_prior = -0.5 * np.sum(w**2, axis=1)
    # Log likelihood: Bernoulli with sigmoid link
    linear_combination = X_data @ w.T
    # Use a numerically stable log-likelihood calculation
    log_likelihood = np.sum(y_data[:, np.newaxis] * linear_combination - np.log(1 + np.exp(linear_combination)), axis=0)
    return log_prior + log_likelihood

def log_unnormalized_p_jax(w):
    # JAX version for HMC, SVGD, GM-ADVI
    log_prior = -0.5 * jnp.sum(w**2)
    linear_combination = jnp.dot(X_train_scaled, w)
    # Numerically stable log-likelihood
    log_likelihood = jnp.sum(y_train * linear_combination - jnp.log(1 + jnp.exp(linear_combination)))
    return log_prior + log_likelihood

def log_unnormalized_p_torch(w):
    # PyTorch version for LMC
    w_torch = torch.tensor(w, dtype=torch.float32, requires_grad=True) if not isinstance(w, torch.Tensor) else w
    X_train_torch = torch.tensor(X_train_scaled, dtype=torch.float32)
    y_train_torch = torch.tensor(y_train, dtype=torch.float32)

    log_prior = -0.5 * torch.sum(w_torch**2)
    linear_combination = torch.mv(X_train_torch, w_torch)
    log_likelihood = torch.sum(y_train_torch * linear_combination - torch.log(1 + torch.exp(linear_combination)))
    return log_prior + log_likelihood

def log_unnormalized_p_pytensor(w):
    # PyTensor version for ADVI
    log_prior = -0.5 * pm.math.sum(w**2)
    linear_combination = pm.math.dot(X_train_scaled, w)
    log_likelihood = pm.math.sum(y_train * linear_combination - pm.math.log(1 + pm.math.exp(linear_combination)))
    return log_prior + log_likelihood


# ##################################################################
# ## GMA SAMPLING (pGD with True Projection) ##
# ##################################################################

# --- GMA Hyperparameters ---
N = 2000  # Total number of Gaussian components
M = 60   # Samples per Gaussian
K = 1500  # Number of iterations for weight updates
eta = 0.1  # Initial learning rate
cov_scale = 0.03 # Use a small covariance for high resolution

# --- Initialize fixed means and covariances ---
# BLR CHANGE: Initialize means from a broad Gaussian in 4D
initial_means = np.random.multivariate_normal(np.zeros(d), 5 * np.eye(d), size=N)
initial_covariances = [cov_scale * np.eye(d) for _ in range(N)]
initial_weights = np.full(N, 1/N)

# --- Initialize samples from each Gaussian component ---
samples = np.zeros((N, M, d))
for i in range(N):
    samples[i] = np.random.multivariate_normal(mean=initial_means[i], cov=initial_covariances[i], size=M)
flat_samples = samples.reshape(N * M, d)

# --- Run GMA ---
def project_to_simplex(v):
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

start_time_GMA = time.time()

# 1. Pre-compute GMM PDFs
pdf_matrix_P = np.zeros((N * M, N))
for l in tqdm(range(N), desc="GMA Pre-computing PDFs"):
    pdf_matrix_P[:, l] = multivariate_normal.pdf(flat_samples, mean=initial_means[l], cov=initial_covariances[l])

# 2. Pre-compute log target densities
log_p_target = log_unnormalized_p(flat_samples, X_train_scaled, y_train)

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
if not np.isclose(np.sum(final_weights), 1.0):
    raise ValueError(f"final_weights: Expected sum to be 1.0, got {np.sum(final_weights)}")
# save weights and gma_samples
np.save('weights.npy', weights)
np.save('gma_samples.npy', gma_samples)


# ##################################################################
# ## BENCHMARKING ALGORITHMS ##
# ##################################################################

### 0. Maximum Likelihood Estimation (MLE)
print("\n--- Running Maximum Likelihood Estimation ---")
mle_model = LogisticRegression(fit_intercept=False, solver='lbfgs')
mle_model.fit(X_train_scaled, y_train)
mle_coefs = mle_model.coef_.flatten()
mle_preds = mle_model.predict(X_test_scaled)
mle_accuracy = accuracy_score(y_test, mle_preds)
print(f"MLE Optimal coefficients: {mle_coefs}")
print(f"MLE Accuracy on the test set: {mle_accuracy:.4f}")


### 1. Metropolis-Hastings
def mh_sampler(n_iter, initial_point=np.zeros(d), proposal_cov=np.eye(d)):
    accepted = []
    current = np.array(initial_point)
    current_log_p = log_unnormalized_p(current, X_train_scaled, y_train)
    for _ in tqdm(range(n_iter), desc="MH Sampling"):
        proposal = current + np.random.multivariate_normal(np.zeros(d), proposal_cov)
        proposal_log_p = log_unnormalized_p(proposal, X_train_scaled, y_train)
        ratio = np.exp(proposal_log_p - current_log_p)
        if np.random.rand() < min(1.0, ratio):
            current = proposal
            current_log_p = proposal_log_p
        accepted.append(current)
    return np.array(accepted)

start_time_mh = time.time()
mh_samples = mh_sampler(n_iter=total_samples, initial_point=np.zeros(d), proposal_cov=0.1*np.eye(d))
end_time_mh = time.time()
mh_time = end_time_mh - start_time_mh
print(f"MH time: {mh_time:.4f} seconds")
# save
np.save('mh_samples.npy', mh_samples)

### 2. Hamiltonian Monte Carlo (NUTS) using Jax
warmup = 1000
step_size = 0.01
num_integration_steps = 20

# BLR CHANGE: Adjust HMC parameters for 4D
hmc = blackjax.hmc(log_unnormalized_p_jax, step_size=step_size, num_integration_steps=num_integration_steps, inverse_mass_matrix=jnp.eye(d))
initial_position = jnp.zeros(d)
state = hmc.init(initial_position)
@jax.jit
def one_step(state, rng_key):
    state, _ = hmc.step(rng_key, state)
    return state, state.position

start_time_hmc = time.time()
samples_hmc = []
keys = jax.random.split(rng_key, total_samples + warmup)
for i in tqdm(range(total_samples + warmup), desc="HMC Sampling"):
    state, sample = one_step(state, keys[i])
    if i >= warmup:
        samples_hmc.append(sample)
hmc_samples = jnp.stack(samples_hmc)
end_time_hmc = time.time()
hmc_time = end_time_hmc - start_time_hmc
print(f"HMC time: {hmc_time:.4f} seconds")
# save
np.save('hmc_samples.npy', hmc_samples)


### 3. Langevin Monte Carlo (LMC)
def langevin_monte_carlo(z0=np.zeros(d), total_steps=2000, lr=1e-2):
    z = torch.tensor(z0, dtype=torch.float32, requires_grad=True)
    samples_lmc = []
    for _ in tqdm(range(total_steps), desc="LMC Sampling"):
        # DEBUGGED: Call the PyTorch version of the log-posterior
        logp = log_unnormalized_p_torch(z)
        grad = torch.autograd.grad(logp, z)[0]
        with torch.no_grad():
            z += lr * grad + torch.randn_like(z) * np.sqrt(2 * lr)
        z.requires_grad_(True)
        samples_lmc.append(z.clone().detach().numpy())
    return np.array(samples_lmc)

start_time_lmc = time.time()
lmc_samples = langevin_monte_carlo(z0=np.zeros(d), total_steps=total_samples, lr=0.01)
end_time_lmc = time.time()
lmc_time = end_time_lmc - start_time_lmc
print(f"LMC time: {lmc_time:.4f} seconds")
# save
np.save('lmc_samples.npy', lmc_samples)


### 4. Stein Variational Gradient Descent (SVGD)
grad_log_p_batch = jax.vmap(jax.grad(log_unnormalized_p_jax))

def rbf_kernel(x, h=None):
    pairwise_diffs = x[:, None, :] - x[None, :, :]
    pairwise_dists_sq = jnp.sum(pairwise_diffs ** 2, axis=-1)
    if h is None:
        med_sq = jnp.median(pairwise_dists_sq)
        h = jnp.sqrt(0.5 * med_sq / jnp.log(x.shape[0] + 1.0))
    K = jnp.exp(-pairwise_dists_sq / (h ** 2))
    grad_K = - (2 / h ** 2) * pairwise_diffs * K[..., None]
    return K, grad_K

@jax.jit
def svgd_step(particles, stepsize):
    grad_logp_vals = grad_log_p_batch(particles)
    K, grad_K = rbf_kernel(particles)
    phi = (K @ grad_logp_vals + jnp.sum(grad_K, axis=1)) / particles.shape[0]
    particles += stepsize * phi
    return particles

def run_svgd(initial_particles, n_iter=500, stepsize=1e-1):
    particles = jnp.array(initial_particles)
    for _ in tqdm(range(n_iter), desc="SVGD Steps"):
        particles = svgd_step(particles, stepsize)
    return np.array(particles)

# Use a fixed number of 800 particles
num_svgd_particles = 3200
stdNormal_initial_particles = np.random.randn(num_svgd_particles, d)
start_time_svgd_std = time.time()
svgd_particles_stdNormal_initial = run_svgd(stdNormal_initial_particles, n_iter=500, stepsize=0.1)
end_time_svgd_std = time.time()
svgd_time_stdNormal_initial = end_time_svgd_std - start_time_svgd_std
print(f"SVGD time (stdNormal initial): {svgd_time_stdNormal_initial:.4f} seconds")
np.save(f'svgd_particles_stdNormal_initial.npy', svgd_particles_stdNormal_initial)

# Randomly sample 800 particles from the initial GMA samples
gma_initial_indices = rng.choice(len(flat_samples), size=num_svgd_particles, replace=False)
GMA_initial_particles = copy.deepcopy(flat_samples)[gma_initial_indices]
start_time_svgd_gma = time.time()
svgd_particles_GMA_initial = run_svgd(GMA_initial_particles, n_iter=500, stepsize=0.1)
end_time_svgd_gma = time.time()
svgd_time_GMA_initial = end_time_svgd_gma - start_time_svgd_gma
print(f"SVGD time (GMA initial): {svgd_time_GMA_initial:.4f} seconds")
np.save(f'svgd_particles_GMA_initial.npy', svgd_particles_GMA_initial)


### 5. MFVI-ADVI (PyMC)
with pm.Model() as advi_model:
    w = pm.MvNormal('w', mu=np.zeros(d), cov=np.eye(d) * 10, shape=d)
    pm.Potential('logp', log_unnormalized_p_pytensor(w))

start_time_advi = time.time()
with advi_model:
    fit_seed = rng.randint(2**32 - 1)
    mean_field_approx = pm.fit(n=30000, method='advi', random_seed=fit_seed)

sample_seed = rng.randint(2**32 - 1)
advi_samples_trace = mean_field_approx.sample(total_samples, random_seed=sample_seed)
advi_samples = advi_samples_trace.posterior['w'].to_numpy().squeeze()
end_time_advi = time.time()
advi_time = end_time_advi - start_time_advi
print(f"ADVI (PyMC) time: {advi_time:.4f} seconds")
# save
np.save(f'advi_samples.npy', advi_samples)


### 6. GM-ADVI (JAX)
start_time_gmadvi = time.time()
K_mix = 100
# T_siwae = 100
# gmadvi_steps = 300
# K_mix = copy.deepcopy(N)  # Number of mixture components in the variational posterior
T_siwae = copy.deepcopy(M)  # Number of stratified samples per component
gmadvi_steps = copy.deepcopy(K)
gmadvi_lr = 1e-2

def gmm_log_prob(z, params):
    log_alphas = jax.nn.log_softmax(params['logits'])
    component_log_probs = jnp.sum(jax.scipy.stats.norm.logpdf(z, loc=params['means'], scale=jnp.exp(params['log_scales'])), axis=-1)
    return jax.scipy.special.logsumexp(log_alphas + component_log_probs)

def siwae_loss(params, key):
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, shape=(T_siwae, K_mix, d))
    z_samples = params['means'][None, :, :] + jnp.exp(params['log_scales'][None, :, :]) * eps
    z_samples_flat = z_samples.reshape(-1, d)
    log_p_flat = jax.vmap(log_unnormalized_p_jax)(z_samples_flat)
    log_p = log_p_flat.reshape(T_siwae, K_mix)
    log_q_flat = jax.vmap(gmm_log_prob, in_axes=(0, None))(z_samples_flat, params)
    log_q = log_q_flat.reshape(T_siwae, K_mix)
    log_alphas = jax.nn.log_softmax(params['logits'])
    log_weights = log_alphas[None, :] + log_p - log_q
    siwae_objective = jax.scipy.special.logsumexp(log_weights, axis=1)
    siwae_objective = jnp.mean(siwae_objective - jnp.log(T_siwae))
    return -siwae_objective

key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(rng_key, 5)
params = {
    'logits': jnp.zeros(K_mix),
    'means': jax.random.uniform(subkey1, shape=(K_mix, d), minval=-5.0, maxval=5.0),
    'log_scales': jnp.log(jax.random.uniform(subkey2, shape=(K_mix, d), minval=0.5, maxval=1.5))
}
optimizer = optax.adam(gmadvi_lr)
opt_state = optimizer.init(params)
@jax.jit
def update_step(params, opt_state, key):
    loss, grads = jax.value_and_grad(siwae_loss)(params, key)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

for step in tqdm(range(gmadvi_steps), desc="GM-ADVI Training"):
    key, subkey = jax.random.split(key)
    params, opt_state, loss = update_step(params, opt_state, subkey)

final_alphas = jax.nn.softmax(params['logits'])
component_choices = jax.random.choice(subkey3, K_mix, shape=(total_samples,), p=final_alphas)
gmadvi_samples = jax.random.normal(subkey4, shape=(total_samples, d)) * jnp.exp(params['log_scales'][component_choices]) + params['means'][component_choices]
gmadvi_samples = np.array(gmadvi_samples)
end_time_gmadvi = time.time()
gmadvi_time = end_time_gmadvi - start_time_gmadvi
print(f"GM-ADVI time: {gmadvi_time:.4f} seconds")
np.save(f'gmadvi_samples.npy', gmadvi_samples)

# Store all execution times
execution_times = {
    'GMA': gma_time, 'MH': mh_time, 'HMC': hmc_time, 'LMC': lmc_time,
    'SVGD (Std init)': svgd_time_stdNormal_initial, 'SVGD (GMA init)': svgd_time_GMA_initial,
    'MFVI-ADVI': advi_time, 'GM-ADVI': gmadvi_time
}
np.save('execution_times.npy', execution_times)

# ##################################################################
# ## EVALUATION AND VISUALIZATION ##
# ##################################################################

all_samples = {
    'GMA': gma_samples, 'MH': mh_samples, 'HMC': np.array(hmc_samples), 'LMC': lmc_samples,
    'SVGD (Std init)': svgd_particles_stdNormal_initial, 'SVGD (GMA init)': svgd_particles_GMA_initial,
    'MFVI-ADVI': advi_samples, 'GM-ADVI': gmadvi_samples
}

# --- Calculate and Print Test Accuracy ---
print("\n--- Test Set Accuracy ---")
# Add MLE to the accuracy report
print(f"MLE: {mle_accuracy:.4f}")
for method, samples_data in all_samples.items():
    # Use the mean of the posterior samples as the point estimate for weights
    w_mean = np.mean(samples_data, axis=0)

    # Predict on the test set
    test_probs = sigmoid(X_test_scaled @ w_mean)
    test_preds = (test_probs > 0.5).astype(int)

    # Calculate accuracy
    accuracy = accuracy_score(y_test, test_preds)
    print(f"{method}: {accuracy:.4f}")

# --- Plot GMA Details ---
# Calculate posterior mean and mode for the GMA samples
gma_w_mean = np.mean(gma_samples, axis=0)
gma_w_mode = np.zeros(d)
for i in range(d):
    counts, bin_edges = np.histogram(gma_samples[:, i], bins=50)
    max_bin_index = np.argmax(counts)
    gma_w_mode[i] = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2

# Create a single figure for the two side-by-side joint plots
fig = plt.figure(figsize=(18, 8))
fig.suptitle(f'GMA Sampling Results (N={N}, K={K}, M={M}, Cov={cov_scale}, eta={eta}, time={gma_time:.2f}s)', fontsize=16)

# Define axis limits based on GMA samples for consistent scaling within this plot
padding_w01 = (gma_samples[:, 0].max() - gma_samples[:, 0].min()) * 0.1
padding_w23 = (gma_samples[:, 2].max() - gma_samples[:, 2].min()) * 0.1
xlim01 = (gma_samples[:, 0].min() - padding_w01, gma_samples[:, 0].max() + padding_w01)
ylim01 = (gma_samples[:, 1].min() - padding_w01, gma_samples[:, 1].max() + padding_w01)
xlim23 = (gma_samples[:, 2].min() - padding_w23, gma_samples[:, 2].max() + padding_w23)
ylim23 = (gma_samples[:, 3].min() - padding_w23, gma_samples[:, 3].max() + padding_w23)

# Create Plot for (w_0, w_1) on the left
gs1 = fig.add_gridspec(2, 2, left=0.05, right=0.48, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
ax1 = fig.add_subplot(gs1[1, 0])
ax1_histx = fig.add_subplot(gs1[0, 0], sharex=ax1)
ax1_histy = fig.add_subplot(gs1[1, 1], sharey=ax1)
ax1_histx.tick_params(axis="x", labelbottom=False)
ax1_histy.tick_params(axis="y", labelleft=False)

ax1.scatter(flat_samples[:, 0], flat_samples[:, 1], alpha=0.05, color='gray', s=5, label='Initial Samples')
ax1.scatter(gma_samples[:, 0], gma_samples[:, 1], alpha=0.2, s=10, label='Ensemble Samples')
ax1.scatter(gma_w_mean[0], gma_w_mean[1], marker='o', s=150, color='orange', zorder=10, label='Posterior Mean')
ax1.scatter(gma_w_mode[0], gma_w_mode[1], marker='X', s=150, color='orange', zorder=10, label='Posterior Mode')
ax1.scatter(mle_coefs[0], mle_coefs[1], marker='P', s=200, color='red', zorder=10, label='MLE')
ax1.set_xlabel('$w_0$')
ax1.set_ylabel('$w_1$')
ax1.legend()
ax1.set_xlim(xlim01)
ax1.set_ylim(ylim01)
ax1_histx.hist(gma_samples[:, 0], bins=30, density=True)
ax1_histy.hist(gma_samples[:, 1], bins=30, orientation='horizontal', density=True)

# Create Plot for (w_2, w_3) on the right
gs2 = fig.add_gridspec(2, 2, left=0.55, right=0.98, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
ax2 = fig.add_subplot(gs2[1, 0])
ax2_histx = fig.add_subplot(gs2[0, 0], sharex=ax2)
ax2_histy = fig.add_subplot(gs2[1, 1], sharey=ax2)
ax2_histx.tick_params(axis="x", labelbottom=False)
ax2_histy.tick_params(axis="y", labelleft=False)

ax2.scatter(flat_samples[:, 2], flat_samples[:, 3], alpha=0.05, color='gray', s=5, label='Initial Samples')
ax2.scatter(gma_samples[:, 2], gma_samples[:, 3], alpha=0.2, s=10, label='Ensemble Samples')
ax2.scatter(gma_w_mean[2], gma_w_mean[3], marker='o', s=150, color='orange', zorder=10, label='Posterior Mean')
ax2.scatter(gma_w_mode[2], gma_w_mode[3], marker='X', s=150, color='orange', zorder=10, label='Posterior Mode')
ax2.scatter(mle_coefs[2], mle_coefs[3], marker='P', s=200, color='red', zorder=10, label='MLE')
ax2.set_xlabel('$w_2$')
ax2.set_ylabel('$w_3$')
ax2.legend()
ax2.set_xlim(xlim23)
ax2.set_ylim(ylim23)
ax2_histx.hist(gma_samples[:, 2], bins=30, density=True)
ax2_histy.hist(gma_samples[:, 3], bins=30, orientation='horizontal', density=True)

plt.show()


# --- Create a separate figure for the weight evolution ---
fig_weights, ax_weights = plt.subplots(1, 1, figsize=(8, 6))
for i in range(N):
    ax_weights.plot(range(K+1), weights[i, :])
ax_weights.set_xlabel('Iteration')
ax_weights.set_ylabel('Weight value')
ax_weights.set_title('Evolution of Component Weights')
ax_weights.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Comparison, Statistics Calculation, and Saving ---

# First, determine global axis limits for consistent scaling
all_w0 = np.concatenate([s[:, 0] for s in all_samples.values()])
all_w1 = np.concatenate([s[:, 1] for s in all_samples.values()])
all_w2 = np.concatenate([s[:, 2] for s in all_samples.values()])
all_w3 = np.concatenate([s[:, 3] for s in all_samples.values()])

padding_w01 = (all_w0.max() - all_w0.min()) * 0.1
padding_w23 = (all_w2.max() - all_w2.min()) * 0.1

xlim01 = (all_w0.min() - padding_w01, all_w0.max() + padding_w01)
ylim01 = (all_w1.min() - padding_w01, all_w1.max() + padding_w01)
xlim23 = (all_w2.min() - padding_w23, all_w2.max() + padding_w23)
ylim23 = (all_w3.min() - padding_w23, all_w3.max() + padding_w23)

# Initialize dictionaries to store statistics
posterior_means = {}
posterior_modes = {}

print("\n--- Posterior Statistics & Visualization ---")
for method, samples_data in all_samples.items():
    # Calculate posterior mean and mode for the current method
    w_mean = np.mean(samples_data, axis=0)
    w_mode = np.zeros(d)
    for i in range(d):
        counts, bin_edges = np.histogram(samples_data[:, i], bins=50)
        max_bin_index = np.argmax(counts)
        w_mode[i] = (bin_edges[max_bin_index] + bin_edges[max_bin_index + 1]) / 2

    # Store and print the statistics
    posterior_means[method] = w_mean
    posterior_modes[method] = w_mode
    print(f"\nMethod: {method}")
    print(f"  Posterior Mean: {np.round(w_mean, 4)}")
    print(f"  Posterior Mode: {np.round(w_mode, 4)}")

    # Create a figure with two sets of axes for the plots
    fig = plt.figure(figsize=(18, 8))
    fig.suptitle(f'{method} - Posterior Marginals (Time: {execution_times[method]:.2f}s)', fontsize=16)

    # --- Create Plot for (w_0, w_1) on the left ---
    gs1 = fig.add_gridspec(2, 2, left=0.05, right=0.48, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
    ax1 = fig.add_subplot(gs1[1, 0])
    ax1_histx = fig.add_subplot(gs1[0, 0], sharex=ax1)
    ax1_histy = fig.add_subplot(gs1[1, 1], sharey=ax1)
    ax1_histx.tick_params(axis="x", labelbottom=False)
    ax1_histy.tick_params(axis="y", labelleft=False)

    ax1.scatter(samples_data[:, 0], samples_data[:, 1], alpha=0.2, s=10)
    ax1.scatter(w_mean[0], w_mean[1], marker='o', s=150, color='orange', zorder=10, label='Posterior Mean')
    ax1.scatter(w_mode[0], w_mode[1], marker='X', s=150, color='orange', zorder=10, label='Posterior Mode')
    ax1.scatter(mle_coefs[0], mle_coefs[1], marker='P', s=200, color='red', zorder=10, label='MLE')
    ax1.set_xlabel('$w_0$')
    ax1.set_ylabel('$w_1$')
    ax1.legend()
    ax1.set_xlim(xlim01)
    ax1.set_ylim(ylim01)
    ax1_histx.hist(samples_data[:, 0], bins=30, density=True)
    ax1_histy.hist(samples_data[:, 1], bins=30, orientation='horizontal', density=True)

    # --- Create Plot for (w_2, w_3) on the right ---
    gs2 = fig.add_gridspec(2, 2, left=0.55, right=0.98, width_ratios=(4, 1), height_ratios=(1, 4), wspace=0.05, hspace=0.05)
    ax2 = fig.add_subplot(gs2[1, 0])
    ax2_histx = fig.add_subplot(gs2[0, 0], sharex=ax2)
    ax2_histy = fig.add_subplot(gs2[1, 1], sharey=ax2)
    ax2_histx.tick_params(axis="x", labelbottom=False)
    ax2_histy.tick_params(axis="y", labelleft=False)

    ax2.scatter(samples_data[:, 2], samples_data[:, 3], alpha=0.2, s=10)
    ax2.scatter(w_mean[2], w_mean[3], marker='o', s=150, color='orange', zorder=10, label='Posterior Mean')
    ax2.scatter(w_mode[2], w_mode[3], marker='X', s=150, color='orange', zorder=10, label='Posterior Mode')
    ax2.scatter(mle_coefs[2], mle_coefs[3], marker='P', s=200, color='red', zorder=10, label='MLE')
    ax2.set_xlabel('$w_2$')
    ax2.set_ylabel('$w_3$')
    ax2.legend()
    ax2.set_xlim(xlim23)
    ax2.set_ylim(ylim23)
    ax2_histx.hist(samples_data[:, 2], bins=30, density=True)
    ax2_histy.hist(samples_data[:, 3], bins=30, orientation='horizontal', density=True)

    plt.show()

# --- Save the collected statistics after the loop ---
np.save('posterior_means.npy', posterior_means)
np.save('posterior_modes.npy', posterior_modes)
print("\nPosterior means and modes saved to .npy files.")

"""# download results to local."""

!zip -r /content/colab_files.zip /content
from google.colab import files
files.download('/content/colab_files.zip')

"""# end."""
