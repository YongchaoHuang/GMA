# -*- coding: utf-8 -*-
"""GMA sampling test 6: Wave [new, used].ipynb

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
N_sqrt = 30 # Create a 30x30 grid of components
N = N_sqrt**2  # Total number of Gaussian components (900)
M = 15  # Samples per Gaussian
K = 800  # Number of iterations for weight updates
eta = 0.1  # Initial learning rate
z_range = (-5, 5) # Support range for initializing component means
cov_scale = 0.02 # Use a very small covariance for high resolution

# --- Initialize fixed means and covariances ---
# 2D CHANGE: Create a grid of means
grid_points = np.linspace(z_range[0], z_range[1], N_sqrt)
xx, yy = np.meshgrid(grid_points, grid_points)
initial_means = np.vstack([xx.ravel(), yy.ravel()]).T

# 2D CHANGE: Initialize covariances as scaled identity matrices
initial_covariances = [cov_scale * np.eye(d) for _ in range(N)]
initial_weights = np.full(N, 1/N)

# --- Initialize samples from each Gaussian component ---
# 2D CHANGE: Samples are now 2D
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
# 2D CHANGE: Use multivariate_normal.pdf
pdf_matrix_P = np.zeros((N * M, N))
for l in range(N):
    pdf_matrix_P[:, l] = multivariate_normal.pdf(flat_samples, mean=initial_means[l], cov=initial_covariances[l])

# 2. Pre-compute log target densities
log_p_target = np.log(unnormalized_p(flat_samples).flatten() + 1e-12)

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
# a robust check for floating-point numbers
if not np.isclose(np.sum(final_weights), 1.0):
    raise ValueError(f"final_weights: Expected sum to be 1.0, got {np.sum(final_weights)}")
# save weights and gma_samples
np.save('weights.npy', weights)
np.save('gma_samples.npy', gma_samples)

# --- Plot GMA Details ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot target density contours
x = np.linspace(z_range[0], z_range[1], 200)
y = np.linspace(z_range[0], z_range[1], 200)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
Z = unnormalized_p(pos)
ax1.contour(X, Y, Z, levels=10, colors='black', alpha=0.6)

# Plot initial and final GMA samples
ax1.scatter(flat_samples[:, 0], flat_samples[:, 1], alpha=0.1, label='Initial samples', color='gray', s=5)
ax1.scatter(gma_samples[:, 0], gma_samples[:, 1], alpha=0.3, label='Ensemble samples', color='green', s=5)
ax1.set_title(f'GMA Results (N={N}, K={K}, M={M}, Cov={cov_scale}, $\eta_0$={eta}, time={np.round(gma_time,2)})')
ax1.set_xlabel('$z_1$')
ax1.set_ylabel('$z_2$')
ax1.legend()
ax1.set_xlim(z_range)
ax1.set_ylim(z_range)
ax1.set_aspect('equal', adjustable='box')

# Plot weight evolution
for i in range(N):
    ax2.plot(range(K+1), weights[i, :])
ax2.hlines(0.2, 0, K+1, linestyles='dashed', colors='red')
ax2.text(50,0.18,f'no. of components with weight>0.2: {np.sum(weights[:,-1] > 0.2)}', fontsize=12)
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Weight value')
ax2.set_title('Evolution of Component Weights')
ax2.grid(True)
plt.tight_layout()
plt.show()

"""## optimal."""

import numpy as np
from scipy.stats import multivariate_normal
from tqdm import tqdm
import matplotlib.pyplot as plt
import time
import torch
import jax
import jax.numpy as jnp
from jax.scipy.stats import multivariate_normal as jax_multivariate_normal
import blackjax
import pymc as pm
import optax
import copy
import warnings

# Suppress verbose output from PyMC
warnings.filterwarnings("ignore", category=FutureWarning)

# Set random seed for reproducibility
np.random.seed(111)
torch.manual_seed(111)
rng_key = jax.random.PRNGKey(111)
rng = np.random.RandomState(111)

# ##################################################################
# ## 2D TARGET DENSITY AND PARAMETERS ##
# ##################################################################
d = 2 # Dimension

# --- WAVE DENSITY: Define the 2D wave-shaped target density ---

def unnormalized_p(z):
    # NumPy version for MH, GMA
    z = np.atleast_2d(z)
    x1 = z[..., 0]
    x2 = z[..., 1]
    return np.exp(-0.5 * (x2 - np.sin(np.pi * x1 / 2)) ** 2 / 0.16)

def log_unnormalized_p_jax(z):
    # JAX version for HMC, SVGD, GM-ADVI
    x1 = z[0]
    x2 = z[1]
    return -0.5 * (x2 - jnp.sin(jnp.pi * x1 / 2)) ** 2 / 0.16

def unnormalized_p_torch(z):
    # PyTorch version for LMC
    x1 = z[0]
    x2 = z[1]
    return torch.exp(-0.5 * (x2 - torch.sin(torch.pi * x1 / 2)) ** 2 / 0.16)

def log_unnormalized_p_pytensor(z):
    # PyTensor version for ADVI
    x1 = z[0]
    x2 = z[1]
    return -0.5 * (x2 - pm.math.sin(np.pi * x1 / 2)) ** 2 / 0.16


# ##################################################################
# ## GMA SAMPLING (pGD with True Projection) ##
# ##################################################################

# --- GMA Hyperparameters ---
N_sqrt = 30 # Create a 30x30 grid of components
N = N_sqrt**2  # Total number of Gaussian components (900)
M = 15  # Samples per Gaussian
K = 800  # Number of iterations for weight updates
eta = 0.1  # Initial learning rate
z_range = (-5, 5) # Support range for initializing component means
cov_scale = 0.02 # Use a very small covariance for high resolution

# --- Initialize fixed means and covariances ---
# 2D CHANGE: Create a grid of means
grid_points = np.linspace(z_range[0], z_range[1], N_sqrt)
xx, yy = np.meshgrid(grid_points, grid_points)
initial_means = np.vstack([xx.ravel(), yy.ravel()]).T

# 2D CHANGE: Initialize covariances as scaled identity matrices
initial_covariances = [cov_scale * np.eye(d) for _ in range(N)]
initial_weights = np.full(N, 1/N)

# --- Initialize samples from each Gaussian component ---
# 2D CHANGE: Samples are now 2D
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
# 2D CHANGE: Use multivariate_normal.pdf
pdf_matrix_P = np.zeros((N * M, N))
for l in range(N):
    pdf_matrix_P[:, l] = multivariate_normal.pdf(flat_samples, mean=initial_means[l], cov=initial_covariances[l])

# 2. Pre-compute log target densities
log_p_target = np.log(unnormalized_p(flat_samples).flatten() + 1e-12)

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
# a robust check for floating-point numbers
if not np.isclose(np.sum(final_weights), 1.0):
    raise ValueError(f"final_weights: Expected sum to be 1.0, got {np.sum(final_weights)}")
# save weights and gma_samples
np.save('weights.npy', weights)
np.save('gma_samples.npy', gma_samples)


# ##################################################################
# ## BENCHMARKING ALGORITHMS ##
# ##################################################################

### 1. Metropolis-Hastings
def mh_sampler(n_iter, initial_point=np.zeros(d), proposal_cov=np.eye(d)):
    # 2D CHANGE: Handle 2D points and proposals
    accepted = []
    current = np.array(initial_point)
    for _ in tqdm(range(n_iter), desc="MH Sampling"):
        proposal = current + np.random.multivariate_normal(np.zeros(d), proposal_cov)
        ratio = unnormalized_p(proposal).flatten()[0] / unnormalized_p(current).flatten()[0]
        if np.random.rand() < min(1.0, ratio):
            current = proposal
        accepted.append(current)
    return np.array(accepted)

start_time_mh = time.time()
mh_samples = mh_sampler(n_iter=total_samples, initial_point=np.zeros(d), proposal_cov=0.05*np.eye(d))
end_time_mh = time.time()
mh_time = end_time_mh - start_time_mh
print(f"MH time: {mh_time:.4f} seconds")
# save
np.save('mh_samples.npy', mh_samples)

### 2. Hamiltonian Monte Carlo (NUTS) using Jax
warmup = 1000
step_size = 0.05
num_integration_steps = 20

# 2D CHANGE: Adjust HMC parameters for 2D
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
    # 2D CHANGE: Use 2D tensors
    z = torch.tensor(z0, dtype=torch.float32, requires_grad=True)
    samples_lmc = []
    for _ in tqdm(range(total_steps), desc="LMC Sampling"):
        logp = torch.log(unnormalized_p_torch(z) + 1e-12)
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
    # This function is already general for d-dimensions
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

# MODIFIED: Use a fixed number of 800 particles
num_svgd_particles = 800
stdNormal_initial_particles = np.random.randn(num_svgd_particles, d)
start_time_svgd_std = time.time()
svgd_particles_stdNormal_initial = run_svgd(stdNormal_initial_particles, n_iter=500, stepsize=0.01)
end_time_svgd_std = time.time()
svgd_time_stdNormal_initial = end_time_svgd_std - start_time_svgd_std
print(f"SVGD time (stdNormal initial): {svgd_time_stdNormal_initial:.4f} seconds")
np.save(f'svgd_particles_stdNormal_initial.npy', svgd_particles_stdNormal_initial)

# MODIFIED: Randomly sample 800 particles from the initial GMA samples
gma_initial_indices = rng.choice(len(flat_samples), size=num_svgd_particles, replace=False)
GMA_initial_particles = copy.deepcopy(flat_samples)[gma_initial_indices]
start_time_svgd_gma = time.time()
svgd_particles_GMA_initial = run_svgd(GMA_initial_particles, n_iter=500, stepsize=0.01)
end_time_svgd_gma = time.time()
svgd_time_GMA_initial = end_time_svgd_gma - start_time_svgd_gma
print(f"SVGD time (GMA initial): {svgd_time_GMA_initial:.4f} seconds")
np.save(f'svgd_particles_GMA_initial.npy', svgd_particles_GMA_initial)


### 5. MFVI-ADVI (PyMC)
# 2D CHANGE: Use a 2D variable in the PyMC model
with pm.Model() as advi_model:
    z = pm.MvNormal('z', mu=np.zeros(d), cov=np.eye(d) * 10, shape=d)
    pm.Potential('logp', log_unnormalized_p_pytensor(z))

start_time_advi = time.time()
with advi_model:
    fit_seed = rng.randint(2**32 - 1)
    mean_field_approx = pm.fit(n=30000, method='advi', random_seed=fit_seed)

sample_seed = rng.randint(2**32 - 1)
advi_samples_trace = mean_field_approx.sample(total_samples, random_seed=sample_seed)
advi_samples = advi_samples_trace.posterior['z'].to_numpy().squeeze()
end_time_advi = time.time()
advi_time = end_time_advi - start_time_advi
print(f"ADVI (PyMC) time: {advi_time:.4f} seconds")
# save
np.save(f'advi_samples.npy', advi_samples)


### 6. GM-ADVI (JAX)
start_time_gmadvi = time.time()
K_mix = copy.deepcopy(N)  # Number of mixture components in the variational posterior
T_siwae = copy.deepcopy(M)  # Number of stratified samples per component
gmadvi_steps = copy.deepcopy(K)
gmadvi_lr = 1e-2

def gmm_log_prob(z, params):
    log_alphas = jax.nn.log_softmax(params['logits'])
    # Diagonal covariance for each component
    component_log_probs = jnp.sum(jax.scipy.stats.norm.logpdf(z, loc=params['means'], scale=params['scales']), axis=-1)
    return jax.scipy.special.logsumexp(log_alphas + component_log_probs)

def siwae_loss(params, key):
    key, subkey = jax.random.split(key)
    # Shape of eps: (T_siwae, K_mix, d)
    eps = jax.random.normal(subkey, shape=(T_siwae, K_mix, d))
    # Shape of z_samples: (T_siwae, K_mix, d)
    z_samples = params['means'][None, :, :] + params['scales'][None, :, :] * eps

    # --- DEBUGGED SECTION ---
    # Flatten samples for simpler, unambiguous vmapping
    z_samples_flat = z_samples.reshape(-1, d) # Shape: (T_siwae * K_mix, d)

    # Calculate log p(z) for all flattened samples and reshape
    log_p_flat = jax.vmap(log_unnormalized_p_jax)(z_samples_flat)
    log_p = log_p_flat.reshape(T_siwae, K_mix)

    # Calculate log q(z) for all flattened samples and reshape
    log_q_flat = jax.vmap(gmm_log_prob, in_axes=(0, None))(z_samples_flat, params)
    log_q = log_q_flat.reshape(T_siwae, K_mix)
    # --- END DEBUGGED SECTION ---

    log_alphas = jax.nn.log_softmax(params['logits'])
    log_weights = log_alphas[None, :] + log_p - log_q
    siwae_objective = jax.scipy.special.logsumexp(log_weights, axis=1)
    siwae_objective = jnp.mean(siwae_objective - jnp.log(T_siwae))
    return -siwae_objective

key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(rng_key, 5)

params = {
    'logits': jnp.zeros(K_mix),
    'means': jax.random.uniform(subkey1, shape=(K_mix, d), minval=-5.0, maxval=5.0),
    'scales': jax.random.uniform(subkey2, shape=(K_mix, d), minval=0.5, maxval=1.5)
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
gmadvi_samples = jax.random.normal(subkey4, shape=(total_samples, d)) * params['scales'][component_choices] + params['means'][component_choices]
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
# ## VISUALIZATION ##
# ##################################################################

# --- Plot GMA Details ---
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Plot target density contours
x = np.linspace(z_range[0], z_range[1], 200)
y = np.linspace(z_range[0], z_range[1], 200)
X, Y = np.meshgrid(x, y)
pos = np.dstack((X, Y))
Z = unnormalized_p(pos)
ax1.contour(X, Y, Z, levels=10, colors='black', alpha=0.6)

# Plot initial and final GMA samples
ax1.scatter(flat_samples[:, 0], flat_samples[:, 1], alpha=0.1, label='Initial samples', color='gray', s=5)
ax1.scatter(gma_samples[:, 0], gma_samples[:, 1], alpha=0.3, label='Ensemble samples', color='green', s=5)
ax1.set_title(f'GMA Results (N={N}, K={K}, M={M}, Cov={cov_scale}, $\eta_0$={eta}, time={np.round(gma_time, 2)})')
ax1.set_xlabel('$z_1$')
ax1.set_ylabel('$z_2$')
ax1.legend()
ax1.set_xlim(z_range)
ax1.set_ylim(z_range)
ax1.set_aspect('equal', adjustable='box')

# Plot weight evolution
for i in range(N):
    ax2.plot(range(K+1), weights[i, :])
ax2.set_xlabel('Iteration')
ax2.set_ylabel('Weight value')
ax2.set_title('Evolution of Component Weights')
ax2.grid(True)
plt.tight_layout()
plt.show()

# --- Plot Comparison of All Methods ---
all_samples = {
    'GMA': gma_samples, 'MH': mh_samples, 'HMC': np.array(hmc_samples), 'LMC': lmc_samples,
    'SVGD (Std init)': svgd_particles_stdNormal_initial, 'SVGD (GMA init)': svgd_particles_GMA_initial,
    'MFVI-ADVI': advi_samples, 'GM-ADVI': gmadvi_samples
}

fig, axes = plt.subplots(2, 4, figsize=(20, 10), sharex=True, sharey=True)
axes = axes.ravel()

for i, (method, samples_data) in enumerate(all_samples.items()):
    ax = axes[i]
    ax.contour(X, Y, Z, levels=10, colors='black', alpha=0.4)
    ax.scatter(samples_data[:, 0], samples_data[:, 1], alpha=0.2, s=5)
    ax.set_title(f"{method}\nTime: {execution_times[method]:.2f}s", fontsize=12)
    ax.set_xlim(z_range)
    ax.set_ylim(z_range)
    ax.set_aspect('equal', adjustable='box')
    if i >= 4:
        ax.set_xlabel('$z_1$')
    if i % 4 == 0:
        ax.set_ylabel('$z_2$')

plt.tight_layout()
plt.show()

"""# download results to local."""

!zip -r /content/colab_files.zip /content
from google.colab import files
files.download('/content/colab_files.zip')

"""# end."""
