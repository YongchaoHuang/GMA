# -*- coding: utf-8 -*-
"""GMA sampling test 1: connected Trimodal bumps [new, used].ipynb

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
"""

import numpy as np
from scipy.stats import norm
from tqdm import tqdm
from scipy.integrate import quad
import matplotlib.pyplot as plt
import time
import torch
import jax
import jax.numpy as jnp
from jax.scipy.stats import norm as jax_norm
import blackjax
import pymc as pm
import optax # New import for JAX optimization
import copy

# Set random seed for reproducibility
np.random.seed(111)
torch.manual_seed(111)
rng_key = jax.random.PRNGKey(111)
rng = np.random.RandomState(111)

# Parameters
N = 10  # number of Gaussian components
M = 200  # Samples per Gaussian
K = 120  # Number of iterations for weight updates
eta = 0.5  # Initial learning rate (eta_0) for the diminishing schedule
theta_range = (-6, 6)  # Support range for initialising the Gaussian means

# Target distribution (unnormalized) - NumPy version for MH
def unnormalized_p(theta):
    b = 0.1  # Curvature parameter
    sigma = 1.0  # Scale parameter
    return np.exp(-((theta**2 + b * theta**4)**2) / (2 * sigma**2)) + \
           0.3 * norm.pdf(theta, loc=3, scale=0.5) + \
           0.2 * norm.pdf(theta, loc=-3, scale=0.6)

# Initialize fixed means and variances, and initial weights
initial_means = np.linspace(theta_range[0], theta_range[1], N)
initial_variances = np.linspace(0.5**2, 0.7**2, N)
initial_weights = np.full(N, 1/N)
print(f'initial_means: {initial_means}')
print(f'initial_variances: {initial_variances}')
print(f'initial_weights: {initial_weights}')

# Initialize samples from each Gaussian component
samples = np.zeros((N, M))
for i in range(N):
    samples[i] = np.random.normal(loc=initial_means[i], scale=np.sqrt(initial_variances[i]), size=M)
flat_samples = samples.flatten()
if len(flat_samples) != N*M:
    raise ValueError(f"samples: Expected {N*M} samples, got {len(flat_samples)}")


# ##################################################################
# ## GMA SAMPLING (pGD with True Projection) ##
# ##################################################################

def project_to_simplex(v):
    """
    Projects a vector v onto the probability simplex using an efficient algorithm.
    This function sorts the values to efficiently find a threshold (theta) which it then subtracts from the original vector to ensure the sum-to-one and non-negativity constraints are met perfectly.
    """
    n_features = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(n_features) + 1
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / float(rho)
    w = np.maximum(v - theta, 0)
    return w

# Start timer for the improved GMA sampling
start_time_GMA = time.time()

# 1. Pre-compute GMM PDFs: Matrix P of shape (NM, N)
pdf_matrix_P = np.zeros((N * M, N))
for l in range(N):
    pdf_matrix_P[:, l] = norm.pdf(flat_samples, loc=initial_means[l], scale=np.sqrt(initial_variances[l]))

# 2. Pre-compute log target densities: Vector log_p_target of shape (NM,)
log_p_target = np.log(unnormalized_p(flat_samples) + 1e-9) # Add small epsilon for numerical stability

# Initialize weights for gradient descent
weights = np.zeros((N, K+1))
weights[:, 0] = initial_weights

# 3. Iteratively update weights using pre-computed values
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
print(f"Improved GMA approximation time: {gma_time:.4f} seconds")

# Ensemble sampling (Stratified with weight proportional selection)
final_weights = weights[:, -1]
selected_indices = rng.choice(N, N*M, p=final_weights, replace=True)
sample_indices = np.random.randint(0, M, size=N*M)
gma_samples = copy.deepcopy(samples)
gma_samples = gma_samples[selected_indices, sample_indices]
if len(samples.flatten()) != N*M:
    raise ValueError(f"samples: Expected {N*M} samples, got {len(samples.flatten())}")
if len(gma_samples.flatten()) != N*M:
    raise ValueError(f"gma_samples: Expected {N*M} samples, got {len(gma_samples.flatten())}")
if sum(final_weights) != 1.0:
    raise ValueError(f"final_weights: Expected sum to be 1.0, got {sum(final_weights)}")

# Compute theoretical approximate density for plotting
theta_grid_fine = np.linspace(theta_range[0], theta_range[1], 200)
approx_density = np.sum([w * norm.pdf(theta_grid_fine, loc=initial_means[l], scale=np.sqrt(initial_variances[l]))
                           for l, w in enumerate(final_weights)], axis=0)

# save weights and gma_samples
np.save('weights.npy', weights)
np.save('gma_samples.npy', gma_samples)

# GMA plots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 5), gridspec_kw={'width_ratios': [3, 3]})
no_of_bins = 60
theta_grid = np.linspace(theta_range[0], theta_range[1], 200)
target_density_vals = unnormalized_p(theta_grid)
normalization_const, _ = quad(unnormalized_p, -20, 20)
target_density = target_density_vals / normalization_const
ax1.hist(samples.flatten(), bins=no_of_bins, density=True, alpha=0.5, color='gray', label='Initial samples')
ax1.hist(gma_samples, bins=no_of_bins, density=True, alpha=0.5, color='green', label='Ensemble samples')
ax1.plot(theta_grid, target_density, label='Target density', color='r')
ax1.plot(theta_grid_fine, approx_density, label='Approximate density', color='green', linestyle='--')
ax1.legend()
ax1.set_title('Density inference')
for i in range(N):
    ax2.plot(range(K+1), weights[i, :], label=f'Weight {i+1}')
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Weight value')
ax2.set_title('Evolution of component weights')
ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True)
plt.tight_layout()
plt.show()


# ##################################################################
# ## BENCHMARKING AND VISUALIZATION ##
# ##################################################################
### 1. Metropolis-Hastings
def mh_sampler(n_iter, initial_point=0.0, proposal_std=1.0):
    accepted = []
    current = np.array(initial_point)
    for _ in range(n_iter):
        proposal = current + np.random.normal(loc=0.0, scale=proposal_std)
        ratio = unnormalized_p(proposal) / unnormalized_p(current)
        if np.random.rand() < min(1.0, ratio):
            current = proposal
        accepted.append(current)
    return np.array(accepted)

start_time_mh = time.time()
mh_samples = mh_sampler(n_iter=N * M, initial_point=0.0, proposal_std=1.0)
end_time_mh = time.time()
mh_time = end_time_mh - start_time_mh
print(f"MH time: {mh_time:.4f} seconds")
print(f'MH samples shape:{mh_samples.shape}')
# save
np.save('mh_samples.npy', mh_samples)
if len(mh_samples.flatten()) != N*M:
    raise ValueError(f"mh_samples: Expected {N*M} samples, got {len(mh_samples.flatten())}")

### 2. Hamiltonian Monte Carlo (NUTS) using Jax
n_samples = N*M
warmup = 1000
step_size = 0.01
num_integration_steps = 20

def log_unnormalized_p_jax(z):
    z = jnp.squeeze(z)
    b = 0.1
    sigma = 1.0
    term1 = jnp.exp(-((z**2 + b * z**4)**2) / (2 * sigma**2))
    term2 = 0.3 * jax_norm.pdf(z, loc=3.0, scale=0.5)
    term3 = 0.2 * jax_norm.pdf(z, loc=-3.0, scale=0.6)
    return jnp.log(term1 + term2 + term3 + 1e-12)

hmc = blackjax.hmc(log_unnormalized_p_jax, step_size=step_size, num_integration_steps=num_integration_steps, inverse_mass_matrix=jnp.array([1.0]))
initial_position = jnp.array(0.0)
state = hmc.init(initial_position)
@jax.jit
def one_step(state, rng_key):
    state, _ = hmc.step(rng_key, state)
    return state, state.position

start_time_hmc = time.time()
samples_hmc = []
keys = jax.random.split(rng_key, n_samples + warmup)
for i in tqdm(range(n_samples + warmup), desc="HMC Sampling"):
    state, sample = one_step(state, keys[i])
    if i >= warmup:
        samples_hmc.append(sample)
hmc_samples = jnp.stack(samples_hmc)
end_time_hmc = time.time()
hmc_time = end_time_hmc - start_time_hmc
print(f"HMC time: {hmc_time:.4f} seconds")
print(f"HMC samples shape: {hmc_samples.shape}")
# save
np.save('hmc_samples.npy', hmc_samples)
if len(hmc_samples.flatten()) != N*M:
    raise ValueError(f"hmc_samples: Expected {N*M} samples, got {len(hmc_samples.flatten())}")

### 3. Langevin Monte Carlo (LMC)
def unnormalized_p_torch(theta):
    b = 0.1
    sigma = 1.0
    term1 = torch.exp(-((theta**2 + b * theta**4)**2) / (2 * sigma**2))
    norm1 = torch.exp(-0.5 * ((theta - 3.0) / 0.5)**2) / (0.5 * (2 * torch.pi)**0.5)
    norm2 = torch.exp(-0.5 * ((theta + 3.0) / 0.6)**2) / (0.6 * (2 * torch.pi)**0.5)
    return term1 + 0.3 * norm1 + 0.2 * norm2

def langevin_monte_carlo(z0=None, total_steps=2000, lr=1e-2, noise_scale=0.02):
    z = torch.tensor(z0 if z0 is not None else 0.0, dtype=torch.float32, requires_grad=True)
    samples_lmc = []
    for _ in tqdm(range(total_steps), desc="LMC Sampling"):
        logp = torch.log(unnormalized_p_torch(z) + 1e-12)
        grad = torch.autograd.grad(logp, z)[0]
        with torch.no_grad():
            z += lr * grad + torch.randn_like(z) * np.sqrt(noise_scale)
        z.requires_grad_(True)
        samples_lmc.append(z.item())
    return np.array(samples_lmc)

start_time_lmc = time.time()
lmc_samples = langevin_monte_carlo(z0=0.0, total_steps=N*M, lr=0.01, noise_scale=0.02)
end_time_lmc = time.time()
lmc_time = end_time_lmc - start_time_lmc
print(f"LMC time: {lmc_time:.4f} seconds")
print(f'LMC samples shape: {lmc_samples.shape}')
# save
np.save('lmc_samples.npy', lmc_samples)
if len(lmc_samples.flatten()) != N*M:
    raise ValueError(f"lmc_samples: Expected {N*M} samples, got {len(lmc_samples.flatten())}")

### 4. Stein Variational Gradient Descent (SVGD)
def log_p_svgd(theta):
    return log_unnormalized_p_jax(theta)

grad_log_p_batch = jax.vmap(jax.grad(log_p_svgd))

def rbf_kernel(x, h=None):
    if x.ndim == 1: x = x[:, None]
    pairwise_diffs = x[:, None, :] - x[None, :, :]
    pairwise_dists_sq = jnp.sum(pairwise_diffs ** 2, axis=-1)
    if h is None:
        med_sq = jnp.median(pairwise_dists_sq)
        h = med_sq / (2 * jnp.log(x.shape[0] + 1.))
    K = jnp.exp(-pairwise_dists_sq / (2*h))
    grad_K = - (1/h) * pairwise_diffs * K[:, :, None]
    return K, grad_K

@jax.jit
def svgd_step(particles, stepsize):
    grad_logp_vals = grad_log_p_batch(particles)
    K, grad_K = rbf_kernel(particles)
    phi = (K @ grad_logp_vals + jnp.sum(grad_K, axis=1)) / particles.shape[0]
    particles += stepsize * phi
    return particles

def run_svgd(initial_particles, n_iter=500, stepsize=1e-2):
    particles = jnp.array(initial_particles)
    if particles.ndim == 1: particles = particles[:, None]
    for _ in tqdm(range(n_iter), desc="SVGD Steps"):
        particles = svgd_step(particles, stepsize)
    return np.array(particles).flatten()

stdNormal_initial_particles = np.random.randn(N*M)
np.save('svgd_stdNormal_initial_particles.npy', stdNormal_initial_particles)
start_time_svgd_std = time.time()
svgd_particles_stdNormal_initial = run_svgd(stdNormal_initial_particles, n_iter=500, stepsize=1e-2)
end_time_svgd_std = time.time()
svgd_time_stdNormal_initial = end_time_svgd_std - start_time_svgd_std
print(f"SVGD time (stdNormal initial): {svgd_time_stdNormal_initial:.4f} seconds")
print(f'SVGD particles shape (stdNormal_initial): {svgd_particles_stdNormal_initial.shape}')
# save
np.save(f'svgd_particles_stdNormal_initial.npy', svgd_particles_stdNormal_initial)
if len(svgd_particles_stdNormal_initial) != N*M:
    raise ValueError(f"SVGD (stdNormal_initial): Expected {N*M} samples, got {len(svgd_particles_stdNormal_initial.flatten())}")

GMA_initial_particles = copy.deepcopy(samples.flatten())
np.save('svgd_GMA_initial_particles.npy', GMA_initial_particles)
start_time_svgd_gma = time.time()
svgd_particles_GMA_initial = run_svgd(GMA_initial_particles, n_iter=500, stepsize=1e-2)
end_time_svgd_gma = time.time()
svgd_time_GMA_initial = end_time_svgd_gma - start_time_svgd_gma
print(f"SVGD time (GMA initial): {svgd_time_GMA_initial:.4f} seconds")
print(f'SVGD particles shape (GMA_initial): {svgd_particles_GMA_initial.shape}')
# save
np.save(f'svgd_particles_GMA_initial.npy', svgd_particles_GMA_initial)
if len(svgd_particles_GMA_initial) != N*M:
    raise ValueError(f"SVGD (GMA_initial): Expected {N*M} samples, got {len(svgd_particles_GMA_initial.flatten())}")


### 5. MFVI-ADVI (Kucukelbir et al.) ###
# This implements Mean-Field Variational Inference using the ADVI algorithm.
# It fits a unimodal Gaussian approximation (mean-field) to the posterior.
def log_unnormalized_p_pytensor(theta):
    b = 0.1
    sigma = 1.0
    term1 = pm.math.exp(-((theta**2 + b * theta**4)**2) / (2 * sigma**2))
    term2 = 0.3 * pm.math.exp(pm.logp(pm.Normal.dist(mu=3.0, sigma=0.5), theta))
    term3 = 0.2 * pm.math.exp(pm.logp(pm.Normal.dist(mu=-3.0, sigma=0.6), theta))
    return pm.math.log(term1 + term2 + term3 + 1e-12)

with pm.Model() as advi_model:
    theta = pm.Uniform('theta', lower=-10.0, upper=10.0)
    pm.Potential('logp', log_unnormalized_p_pytensor(theta))

start_time_advi = time.time()
with advi_model:
    fit_seed = rng.randint(2**32 - 1)
    mean_field_approx = pm.fit(n=copy.deepcopy(K), method='advi', random_seed=fit_seed)

sample_seed = rng.randint(2**32 - 1)
advi_samples_trace = mean_field_approx.sample(N*M, random_seed=sample_seed)
advi_samples = advi_samples_trace.posterior['theta'].to_numpy().flatten()
end_time_advi = time.time()
advi_time = end_time_advi - start_time_advi
print(f"ADVI (PyMC) time: {advi_time:.4f} seconds")
print(f'ADVI (PyMC) samples shape: {advi_samples.shape}')
# save
np.save(f'advi_samples.npy', advi_samples)
if len(advi_samples) != N*M:
    raise ValueError(f"ADVI (PyMC): Expected {N*M} samples, got {len(advi_samples.flatten())}")


### 6. GM-ADVI (Morningstar et al.) ###
# This is a from-scratch implementation of ADVI with a Gaussian Mixture posterior,
# optimized with the SIWAE objective as described in the paper.
start_time_gmadvi = time.time()

# --- Hyperparameters for GM-ADVI ---
K_mix = copy.deepcopy(N)  # Number of mixture components in the variational posterior
T_siwae = copy.deepcopy(M)  # Number of stratified samples per component
gmadvi_steps = copy.deepcopy(K)
gmadvi_lr = 1e-3

# --- (1). Define the GMM posterior and SIWAE loss ---
def gmm_log_prob(z, params):
    log_alphas = jax.nn.log_softmax(params['logits'])
    component_log_probs = jax_norm.logpdf(z, loc=params['means'], scale=params['scales'])
    return jax.scipy.special.logsumexp(log_alphas + component_log_probs)

def siwae_loss(params, key):
    # Stratified sampling: T samples from each of K_mix components
    # Shape of z_samples: (T_siwae, K_mix)
    key, subkey = jax.random.split(key)
    eps = jax.random.normal(subkey, shape=(T_siwae, K_mix))
    z_samples = params['means'] + params['scales'] * eps

    # Calculate log p(z) for all samples (target distribution)
    log_p = log_unnormalized_p_jax(z_samples)

    # Calculate log q(z) for all samples (variational posterior)
    # vmap to compute log_prob for each sample efficiently
    log_q = jax.vmap(jax.vmap(gmm_log_prob, in_axes=(0, None)), in_axes=(0, None))(z_samples, params)

    # Calculate log mixture weights
    log_alphas = jax.nn.log_softmax(params['logits'])

    # SIWAE objective calculation
    log_weights = log_alphas + log_p - log_q
    siwae_objective = jax.scipy.special.logsumexp(log_weights, axis=1) # Sum over K_mix
    siwae_objective = jnp.mean(siwae_objective - jnp.log(T_siwae)) # Average over T_siwae

    # We want to maximize the objective, so we return its negative for minimization
    return -siwae_objective

# --- (2). Set up the optimization ---
key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(rng_key, 5)

# Initialize GMM parameters
params = {
    'logits': jnp.zeros(K_mix),
    'means': jax.random.uniform(subkey1, shape=(K_mix,), minval=-5.0, maxval=5.0),
    'scales': jax.random.uniform(subkey2, shape=(K_mix,), minval=0.5, maxval=1.5)
}

optimizer = optax.adam(gmadvi_lr)
opt_state = optimizer.init(params)

# Define the update step
@jax.jit
def update_step(params, opt_state, key):
    loss, grads = jax.value_and_grad(siwae_loss)(params, key)
    updates, opt_state = optimizer.update(grads, opt_state)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

# --- (3). Run the optimization loop ---
for step in tqdm(range(gmadvi_steps), desc="GM-ADVI Training"):
    key, subkey = jax.random.split(key)
    params, opt_state, loss = update_step(params, opt_state, subkey)

# --- (4). Extract final samples from the learned GMM ---
final_alphas = jax.nn.softmax(params['logits'])
component_choices = jax.random.choice(subkey3, K_mix, shape=(N*M,), p=final_alphas)
gmadvi_samples = jax.random.normal(subkey4, shape=(N*M,)) * params['scales'][component_choices] + params['means'][component_choices]
gmadvi_samples = np.array(gmadvi_samples)

end_time_gmadvi = time.time()
gmadvi_time = end_time_gmadvi - start_time_gmadvi
print(f"GM-ADVI time: {gmadvi_time:.4f} seconds")
print(f'GM-ADVI samples shape: {gmadvi_samples.shape}')
# save
np.save(f'gmadvi_samples.npy', gmadvi_samples)
if len(gmadvi_samples) != N*M:
    raise ValueError(f"GM-ADVI: Expected {N*M} samples, got {len(gmadvi_samples.flatten())}")


#### Final Visualization of all methods
fig, ax1 = plt.subplots(1, 1, figsize=(12, 7))
ax1.hist(gma_samples, bins=no_of_bins, density=True, alpha=0.6, label='GMA (pGD)')
ax1.hist(mh_samples, bins=no_of_bins, density=True, alpha=0.5, histtype='step', linewidth=2, label='MH')
ax1.hist(np.array(hmc_samples), bins=no_of_bins, density=True, alpha=0.5, histtype='step', linewidth=2, label='HMC')
ax1.hist(lmc_samples, bins=no_of_bins, density=True, alpha=0.5, histtype='step', linewidth=2, label='LMC')
ax1.hist(svgd_particles_GMA_initial, bins=no_of_bins, density=True, alpha=0.5, histtype='step', linewidth=2, label='SVGD (GMA init)')
ax1.hist(advi_samples, bins=no_of_bins, density=True, alpha=0.5, histtype='step', linewidth=2, label='ADVI')
ax1.hist(gmadvi_samples, bins=no_of_bins, density=True, alpha=0.5, histtype='step', linewidth=2, label='GM-ADVI')
ax1.plot(theta_grid, target_density, label='Target Density', color='r', linewidth=2)
ax1.plot(theta_grid_fine, approx_density, label='GMA Approx. Density', color='g', linestyle='--', linewidth=2)
ax1.legend()
ax1.set_title('Comparison of Sampling Methods')
ax1.set_xlabel('$\\theta$')
ax1.set_ylabel('Density')
ax1.set_xlim(theta_range)
plt.show()

# Save execution times and hyperparameters
execution_times = {
    'gma_time': gma_time, 'mh_time': mh_time, 'hmc_time': hmc_time, 'lmc_time': lmc_time,
    'svgd_time_stdNormal_initial': svgd_time_stdNormal_initial, 'svgd_time_GMA_initial': svgd_time_GMA_initial,
    'advi_time': advi_time, 'gmadvi_time': gmadvi_time
}
np.save('execution_times.npy', execution_times)
hyper_params = {
    'N': N, 'M': M, 'K': K, 'eta': eta, 'theta_range': theta_range, 'gma_initial_weights': initial_weights,
    'mh_n_iter': N*M, 'hmc_n_samples': n_samples, 'hmc_warmup': warmup, 'hmc_step_size': step_size,
    'hmc_num_integration_steps': num_integration_steps, 'lmc_total_steps': N*M, 'lmc_lr': 0.01,
    'lmc_noise_scale': 0.02, 'svgd_n_iter': 500, 'svgd_stepsize': 1e-2,
    'advi_fit_n': K, 'gmadvi_steps': gmadvi_steps, 'gmadvi_K_mix': K_mix, 'gmadvi_T_siwae': T_siwae
}
np.save('hyper_parameters.npy', hyper_params)

"""# download results to local."""

!zip -r /content/colab_files.zip /content
from google.colab import files
files.download('/content/colab_files.zip')

"""# end."""
