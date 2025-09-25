# -*- coding: utf-8 -*-
"""GMA results analysis [new, used].ipynb

#yongchao.huang@abdn.ac.uk
"""

import numpy as np
from scipy.stats import norm, wasserstein_distance, ks_2samp, entropy
from scipy.integrate import quad
from scipy.spatial.distance import cdist
import matplotlib.pyplot as plt
import copy

# ##################################################################
# ## LOAD SAMPLES, WEIGHTS, AND EXECUTION TIMES ##
# ##################################################################
print("Loading results from .npy files...")

# Load all samples
mh_samples = np.load('mh_samples.npy', allow_pickle=True)
gma_samples = np.load('gma_samples.npy', allow_pickle=True)
svgd_particles_GMA_initial = np.load('svgd_particles_GMA_initial.npy', allow_pickle=True)
svgd_particles_stdNormal_initial = np.load('svgd_particles_stdNormal_initial.npy', allow_pickle=True)
hmc_samples = np.load('hmc_samples.npy', allow_pickle=True)
lmc_samples = np.load('lmc_samples.npy', allow_pickle=True)
# ### ADDED: Load new method samples ###
advi_samples = np.load('advi_samples.npy', allow_pickle=True)
gmadvi_samples = np.load('gmadvi_samples.npy', allow_pickle=True)

# Load weight trajectories for GMA
weights = np.load('weights.npy', allow_pickle=True)
# Load execution times
execution_times = np.load('execution_times.npy', allow_pickle=True).item()
print("All files loaded successfully.")

# ##################################################################
# ## INFERRED DENSITY COMPARISON ##
# ##################################################################

# --- Target Distribution Definition (for plotting) ---
N=10
theta_range = (-6, 6)
# # connected 3 bumps
# def unnormalized_p(theta):
#     b = 0.1
#     sigma = 1.0
#     return np.exp(-((theta**2 + b * theta**4)**2) / (2 * sigma**2)) + \
#            0.3 * norm.pdf(theta, loc=3, scale=0.5) + \
#            0.2 * norm.pdf(theta, loc=-3, scale=0.6)
# separated 3 bumps
def unnormalized_p(theta):
    b = 0.1  # Curvature parameter
    sigma = 1.0  # Scale parameter
    return np.exp(-((theta**2 + b * theta**4)**2) / (2 * sigma**2)) + \
           0.3 * norm.pdf(theta, loc=5, scale=0.2) + \
           0.2 * norm.pdf(theta, loc=-5, scale=0.2)

# --- Compute densities for plotting ---
final_weights = weights[:, -1]
initial_means = np.linspace(theta_range[0], theta_range[1], N)
initial_variances = np.linspace(0.5**2, 0.7**2, N)
theta_grid_fine = np.linspace(theta_range[0], theta_range[1], 200)
approx_density = np.sum([w * norm.pdf(theta_grid_fine, loc=initial_means[l], scale=np.sqrt(initial_variances[l]))
                           for l, w in enumerate(final_weights)], axis=0)

no_of_bins = 60
theta_grid = np.linspace(theta_range[0], theta_range[1], 200)
target_density_vals = unnormalized_p(theta_grid)
normalization_const, _ = quad(unnormalized_p, -20, 20)
target_density = target_density_vals / normalization_const

# --- Load initial particles for SVGD plots ---
samples = np.load('svgd_GMA_initial_particles.npy', allow_pickle=True)
GMA_initial_particles = copy.deepcopy(samples)
stdNormal_initial_particles = np.load('svgd_stdNormal_initial_particles.npy', allow_pickle=True)

# --- Create plots for each method ---
### MODIFIED: Increased number of subplots to 8 ###
fig, axs = plt.subplots(1, 8, figsize=(32, 4), sharey=True)

# GMA
axs[0].hist(gma_samples.flatten(), bins=no_of_bins, density=True, alpha=0.5, color='steelblue', label='GMA samples')
axs[0].plot(theta_grid, target_density, label='Target density', color='r')
axs[0].plot(theta_grid_fine, approx_density, label='GMA Approximate', color='green', linestyle='--')
axs[0].hist(samples.flatten(), bins=no_of_bins, density=True, alpha=0.5, color='gray', label='Initial samples')
axs[0].legend()
axs[0].set_title('GMA')
axs[0].grid(True)

# MH
axs[1].hist(mh_samples, bins=no_of_bins, density=True, alpha=0.5, color='coral', label='MH samples')
axs[1].plot(theta_grid, target_density, label='Target density', color='r')
axs[1].scatter([0.0], [0.01], color='black', marker='o', s=10, label='Initial position')
axs[1].legend()
axs[1].set_title('MH')
axs[1].grid(True)

# HMC
axs[2].hist(hmc_samples, bins=no_of_bins, density=True, alpha=0.5, color='forestgreen', label='HMC samples')
axs[2].plot(theta_grid, target_density, label='Target density', color='r')
axs[2].scatter([0.0], [0.01], color='black', marker='o', s=10, label='Initial position')
axs[2].legend()
axs[2].set_title('HMC')
axs[2].grid(True)

# LMC
axs[3].hist(lmc_samples, bins=no_of_bins, density=True, alpha=0.5, color='indianred', label='LMC samples')
axs[3].plot(theta_grid, target_density, label='Target density', color='r')
axs[3].scatter([0.0], [0.01], color='black', marker='o', s=10, label='Initial position')
axs[3].legend()
axs[3].set_title('LMC')
axs[3].grid(True)

# SVGD1 (GMA initial)
axs[4].hist(svgd_particles_GMA_initial, bins=no_of_bins, density=True, alpha=0.5, color='purple', label='SVGD (GMA init)')
axs[4].plot(theta_grid, target_density, label='Target density', color='r')
axs[4].hist(GMA_initial_particles, bins=no_of_bins, density=True, alpha=0.5, color='gray', label='Initial samples')
axs[4].legend()
axs[4].set_title('SVGD (GMA init)')
axs[4].grid(True)

# SVGD2 (stdNormal_initial)
axs[5].hist(svgd_particles_stdNormal_initial, bins=no_of_bins, density=True, alpha=0.5, color='darkviolet', label='SVGD (Std init)')
axs[5].plot(theta_grid, target_density, label='Target density', color='r')
axs[5].hist(stdNormal_initial_particles, bins=no_of_bins, density=True, alpha=0.5, color='gray', label='Initial samples')
axs[5].legend()
axs[5].set_title('SVGD (Std init)')
axs[5].grid(True)

# ### ADDED: ADVI Plot ###
axs[6].hist(advi_samples, bins=no_of_bins, density=True, alpha=0.5, color='gold', label='ADVI samples')
axs[6].plot(theta_grid, target_density, label='Target density', color='r')
axs[6].legend()
axs[6].set_title('ADVI')
axs[6].grid(True)

# ### ADDED: GM-ADVI Plot ###
axs[7].hist(gmadvi_samples, bins=no_of_bins, density=True, alpha=0.5, color='hotpink', label='GM-ADVI samples')
axs[7].plot(theta_grid, target_density, label='Target density', color='r')
axs[7].legend()
axs[7].set_title('GM-ADVI')
axs[7].grid(True)

plt.ylim(0, 0.8)
plt.tight_layout()
plt.show()

# ##################################################################
# ## COMPUTE DISTANCES ##
# ##################################################################
print("\\nCalculating performance metrics...")
# --- 1. Wasserstein Distance ---
wd_gma = wasserstein_distance(mh_samples, gma_samples)
wd_mh = wasserstein_distance(mh_samples, mh_samples)
wd_hmc = wasserstein_distance(mh_samples, hmc_samples)
wd_lmc = wasserstein_distance(mh_samples, lmc_samples)
wd_svgd_GMA_initial = wasserstein_distance(mh_samples, svgd_particles_GMA_initial)
wd_svgd_stdNormal_initial = wasserstein_distance(mh_samples, svgd_particles_stdNormal_initial)
### ADDED ###
wd_advi = wasserstein_distance(mh_samples, advi_samples)
wd_gmadvi = wasserstein_distance(mh_samples, gmadvi_samples)

# --- 2. KS statistic ---
ks_gma, _ = ks_2samp(mh_samples, gma_samples)
ks_mh, _ = ks_2samp(mh_samples, mh_samples)
ks_hmc, _ = ks_2samp(mh_samples, hmc_samples)
ks_lmc, _ = ks_2samp(mh_samples, lmc_samples)
ks_svgd_GMA_initial, _ = ks_2samp(mh_samples, svgd_particles_GMA_initial)
ks_svgd_stdNormal_initial, _ = ks_2samp(mh_samples, svgd_particles_stdNormal_initial)
### ADDED ###
ks_advi, _ = ks_2samp(mh_samples, advi_samples)
ks_gmadvi, _ = ks_2samp(mh_samples, gmadvi_samples)

# --- 3. MMD^2 ---
def rbf_kernel(x, y, sigma=1.0):
    pairwise_dists = cdist(x[:, np.newaxis], y[:, np.newaxis], 'sqeuclidean')
    return np.exp(-pairwise_dists / (2 * sigma**2))
def compute_biased_mmd_sq(x, y, sigma=1.0):
    k_xx = rbf_kernel(x, x, sigma)
    k_yy = rbf_kernel(y, y, sigma)
    k_xy = rbf_kernel(x, y, sigma)
    return k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()

mmd_mh = compute_biased_mmd_sq(mh_samples, mh_samples)
mmd_gma = compute_biased_mmd_sq(mh_samples, gma_samples)
mmd_svgd_GMA_initial = compute_biased_mmd_sq(mh_samples, svgd_particles_GMA_initial)
mmd_svgd_stdNormal_initial = compute_biased_mmd_sq(mh_samples, svgd_particles_stdNormal_initial)
mmd_hmc = compute_biased_mmd_sq(mh_samples, hmc_samples)
mmd_lmc = compute_biased_mmd_sq(mh_samples, lmc_samples)
### ADDED ###
mmd_advi = compute_biased_mmd_sq(mh_samples, advi_samples)
mmd_gmadvi = compute_biased_mmd_sq(mh_samples, gmadvi_samples)

# --- 4. Total Variation Distance ---
def total_variation_distance(samples1, samples2, bins=60, range=None):
    hist1, bin_edges = np.histogram(samples1, bins=bins, range=range, density=True)
    hist2, _ = np.histogram(samples2, bins=bin_edges, density=True)
    bin_widths = np.diff(bin_edges)
    return 0.5 * np.sum(np.abs(hist1 - hist2) * bin_widths)

tv_GMA = total_variation_distance(mh_samples, gma_samples)
tv_MH = total_variation_distance(mh_samples, mh_samples)
tv_HMC = total_variation_distance(mh_samples, hmc_samples)
tv_LMC = total_variation_distance(mh_samples, lmc_samples)
tv_svgd_GMA_initial = total_variation_distance(mh_samples, svgd_particles_GMA_initial)
tv_svgd_stdNormal_initial = total_variation_distance(mh_samples, svgd_particles_stdNormal_initial)
### ADDED ###
tv_advi = total_variation_distance(mh_samples, advi_samples)
tv_gmadvi = total_variation_distance(mh_samples, gmadvi_samples)

# --- 5. KL Divergence ---
def kl_divergence_from_samples(samples1, samples2, n_bins=60):
    min_val, max_val = min(np.min(samples1), np.min(samples2)), max(np.max(samples1), np.max(samples2))
    bins = np.linspace(min_val, max_val, n_bins + 1)
    hist1, _ = np.histogram(samples1, bins=bins, density=False)
    hist2, _ = np.histogram(samples2, bins=bins, density=False)
    pmf1, pmf2 = hist1 / len(samples1), hist2 / len(samples2)
    epsilon = 1e-10
    pmf1_smoothed, pmf2_smoothed = pmf1 + epsilon, pmf2 + epsilon
    pmf1_smoothed /= np.sum(pmf1_smoothed)
    pmf2_smoothed /= np.sum(pmf2_smoothed)
    return entropy(pk=pmf1_smoothed, qk=pmf2_smoothed)

kl_gma = kl_divergence_from_samples(gma_samples, mh_samples)
kl_mh = kl_divergence_from_samples(mh_samples, mh_samples)
kl_hmc = kl_divergence_from_samples(hmc_samples, mh_samples)
kl_lmc = kl_divergence_from_samples(lmc_samples, mh_samples)
kl_svgd_GMA_initial = kl_divergence_from_samples(svgd_particles_GMA_initial, mh_samples)
kl_svgd_stdNormal_initial = kl_divergence_from_samples(svgd_particles_stdNormal_initial, mh_samples)
### ADDED ###
kl_advi = kl_divergence_from_samples(advi_samples, mh_samples)
kl_gmadvi = kl_divergence_from_samples(gmadvi_samples, mh_samples)

print("Metrics calculation complete.")

# ##################################################################
# ## DISTANCES COMPARISON PLOT ##
# ##################################################################

### MODIFIED: Added new methods to all lists ###
methods = ['GMA', 'MH', 'HMC', 'LMC', 'SVGD (GMA)', 'SVGD (Std)', 'ADVI', 'GM-ADVI']
colors = ['steelblue', 'coral', 'forestgreen', 'indianred', 'purple', 'darkviolet', 'gold', 'hotpink']

wd_values = [wd_gma, wd_mh, wd_hmc, wd_lmc, wd_svgd_GMA_initial, wd_svgd_stdNormal_initial, wd_advi, wd_gmadvi]
ks_values = [ks_gma, ks_mh, ks_hmc, ks_lmc, ks_svgd_GMA_initial, ks_svgd_stdNormal_initial, ks_advi, ks_gmadvi]
mmd_values = [mmd_gma, mmd_mh, mmd_hmc, mmd_lmc, mmd_svgd_GMA_initial, mmd_svgd_stdNormal_initial, mmd_advi, mmd_gmadvi]
tv_values = [tv_GMA, tv_MH, tv_HMC, tv_LMC, tv_svgd_GMA_initial, tv_svgd_stdNormal_initial, tv_advi, tv_gmadvi]
kl_values = [kl_gma, kl_mh, kl_hmc, kl_lmc, kl_svgd_GMA_initial, kl_svgd_stdNormal_initial, kl_advi, kl_gmadvi]
time_values = [
    execution_times['gma_time'], execution_times['mh_time'], execution_times['hmc_time'],
    execution_times['lmc_time'], execution_times['svgd_time_GMA_initial'],
    execution_times['svgd_time_stdNormal_initial'], execution_times['advi_time'], execution_times['gmadvi_time']
]

fig, axs = plt.subplots(1, 6, figsize=(24, 5))

# Plot Wasserstein distances
axs[0].bar(methods, wd_values, color=colors)
axs[0].set_title('1D Wasserstein ▼')
axs[0].set_ylabel('Distance')
axs[0].tick_params(axis='x', rotation=90)
axs[0].grid(True, linestyle='--', alpha=0.7)

# Plot KS statistic
axs[1].bar(methods, ks_values, color=colors)
axs[1].set_title('KS statistic ▼')
axs[1].tick_params(axis='x', rotation=90)
axs[1].grid(True, linestyle='--', alpha=0.7)

# Plot MMD
axs[2].bar(methods, mmd_values, color=colors)
axs[2].set_title(r'$MMD^2$ ▼')
axs[2].tick_params(axis='x', rotation=90)
axs[2].grid(True, linestyle='--', alpha=0.7)

# Plot total variation distances
axs[3].bar(methods, tv_values, color=colors)
axs[3].set_title('Total variation ▼')
axs[3].tick_params(axis='x', rotation=90)
axs[3].grid(True, linestyle='--', alpha=0.7)

# Plot KL divergences
axs[4].bar(methods, kl_values, color=colors)
axs[4].set_title('KL divergence ▼')
axs[4].tick_params(axis='x', rotation=90)
axs[4].grid(True, linestyle='--', alpha=0.7)

# Plot execution times
axs[5].bar(methods, time_values, color=colors)
axs[5].set_title('Execution time (s) ▼')
axs[5].tick_params(axis='x', rotation=90)
axs[5].grid(True, linestyle='--', alpha=0.7)

plt.tight_layout()
plt.show()

# ##################################################################
# ## PRINT AND SAVE ALL RESULTS ##
# ##################################################################
metrics_results = {
    'Wasserstein Distance': wd_values, 'KS Statistic': ks_values,
    'MMD': mmd_values, 'Total Variation': tv_values,
    'KL Divergence': kl_values, 'Execution Time': time_values
}
np.save('metrics_results.npy', metrics_results)

print("\\n--- All Metrics ---")
for metric_name, values in metrics_results.items():
    print(f"\\n{metric_name}:")
    for i, value in enumerate(values):
        print(f"    {methods[i]}: {value:.4f}")
