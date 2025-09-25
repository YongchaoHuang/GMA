# -*- coding: utf-8 -*-
"""GMA sampling test 13: LV system.ipynb

#yongchao.huang@abdn.ac.uk
"""

!lscpu

import psutil

print("CPU cores:", psutil.cpu_count(logical=False))
print("Logical CPUs:", psutil.cpu_count(logical=True))
print("Memory (GB):", round(psutil.virtual_memory().total / 1e9, 2))
print("Disk space (GB):", round(psutil.disk_usage('/').total / 1e9, 2))

"""# Begin."""

# a = alpha
# b = beta
# c = delta
# d = gamma
# X: hare
# Y: lynx

"""# pGD-GMA."""

# =========================================================
# LV + GMA (refined, minimal changes) + diagnostics
#   - RK4 (dt=0.005)
#   - Log-sum-exp mixture evaluation with precomputed log-PDFs
#   - Anisotropic diagonal covariances per dimension (tighter)
#   - 2-stage refine (coarse -> focused around top-weight comps)
#   - Two separate sigmas (hare, lynx), Stan-matching priors
#   - NEW: ground-truth lines; hist overlays from bank1 & bank2;
#          1x4 2-D scatter panel; weight-evolution plots
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import log, pi
import time

rng = np.random.default_rng(123)

# ----------------------------
# Data
# ----------------------------
years = np.array([1900,1901,1902,1903,1904,1905,1906,1907,1908,1909,
                  1910,1911,1912,1913,1914,1915,1916,1917,1918,1919,1920], dtype=np.int64)
lynx = np.array([4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1,
                 7.4, 8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6], dtype=np.float64)
hare = np.array([30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4,
                 27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7], dtype=np.float64)

t_obs = (years - years[0]).astype(float)  # 0..20
dt_rk = 0.005                             # smaller RK4 step for fidelity

# Reference (ground-truth) values used for vertical lines
ref_vals = {
    "alpha": 0.55,
    "beta": 0.028,
    "delta": 0.024,
    "gamma": 0.80,
    "X0": 33.956,
    "Y0": 5.933,
    "sigma1": 0.25,
    "sigma2": 0.25,
}

# --------------------------------------
# LV dynamics + RK4 integrator
# --------------------------------------
def lv_rhs(state, p):
    X, Y = state
    a, b, c, d = p
    return np.array([a*X - b*X*Y, c*X*Y - d*Y], dtype=np.float64)

def rk4_integrate(p, X0, Y0, t_eval, dt=dt_rk):
    assert t_eval[0] == 0.0
    steps_total = int(np.ceil((t_eval[-1] - 0.0) / dt))
    state = np.array([X0, Y0], dtype=np.float64)
    t_curr = 0.0
    out = np.empty((len(t_eval), 2), dtype=np.float64)
    next_idx = 0
    out[next_idx] = state
    next_idx += 1
    for _ in range(steps_total):
        if next_idx >= len(t_eval):
            break
        k1 = lv_rhs(state, p)
        k2 = lv_rhs(state + 0.5*dt*k1, p)
        k3 = lv_rhs(state + 0.5*dt*k2, p)
        k4 = lv_rhs(state + dt*k3, p)
        state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        state = np.maximum(state, 1e-12)  # keep positive
        t_curr += dt
        while next_idx < len(t_eval) and t_curr + 1e-12 >= t_eval[next_idx]:
            out[next_idx] = state
            next_idx += 1
    return out

# --------------------------------------
# Parameterization / priors / likelihood
# theta = [log a, log b, log c, log d, log X0, log Y0, log sigma1, log sigma2]
# --------------------------------------
def unpack_theta(theta):
    la, lb, lc, ld, lX0, lY0, ls1, ls2 = theta
    a, b, c, d = np.exp([la, lb, lc, ld])
    X0, Y0 = np.exp([lX0, lY0])
    s1, s2 = np.exp([ls1, ls2])
    return a, b, c, d, X0, Y0, s1, s2

def simulate_from_theta(theta):
    a, b, c, d, X0, Y0, _, _ = unpack_theta(theta)
    traj = rk4_integrate([a, b, c, d], X0, Y0, t_obs)
    return traj[:, 0], traj[:, 1]

def log_normal_pdf(x, mu, sd):
    z = (x - mu) / sd
    return -0.5*z*z - log(sd) - 0.5*log(2*pi)

def log_prior(theta):
    la, lb, lc, ld, lX0, lY0, ls1, ls2 = theta
    a, b, c, d = np.exp([la, lb, lc, ld])
    if not np.isfinite(a*b*c*d) or (a<=0 or b<=0 or c<=0 or d<=0):
        return -np.inf
    lp = 0.0
    # natural-scale Normals (+ Jacobian terms)
    lp += log_normal_pdf(a, 1.0, 0.5) + la       # alpha
    lp += log_normal_pdf(d, 1.0, 0.5) + ld       # gamma
    lp += log_normal_pdf(b, 0.05, 0.05) + lb     # beta
    lp += log_normal_pdf(c, 0.05, 0.05) + lc     # delta
    # LogNormals (Normal on logs)
    mu_lN = np.log(10.0)
    lp += log_normal_pdf(lX0, mu_lN, 1.0)        # log X0
    lp += log_normal_pdf(lY0, mu_lN, 1.0)        # log Y0
    lp += log_normal_pdf(ls1, -1.0, 1.0)         # log sigma1
    lp += log_normal_pdf(ls2, -1.0, 1.0)         # log sigma2
    return lp

def loglik_lognormal(theta):
    a, b, c, d, X0, Y0, s1, s2 = unpack_theta(theta)
    if (s1 <= 1e-10) or (s2 <= 1e-10) or (not np.isfinite(s1)) or (not np.isfinite(s2)):
        return -np.inf
    Xp, Yp = simulate_from_theta(theta)
    Xp = np.maximum(Xp, 1e-12)
    Yp = np.maximum(Yp, 1e-12)
    rx = np.log(hare) - np.log(Xp)
    ry = np.log(lynx) - np.log(Yp)
    nx, ny = rx.size, ry.size
    return (
        -0.5*np.sum((rx*rx)/(s1*s1)) - nx*np.log(s1) - 0.5*nx*np.log(2*np.pi)
        -0.5*np.sum((ry*ry)/(s2*s2)) - ny*np.log(s2) - 0.5*ny*np.log(2*np.pi)
    )

def log_unnormalized_p_thetas(thetas):
    out = np.empty(thetas.shape[0], dtype=np.float64)
    for i in range(thetas.shape[0]):
        lp = log_prior(thetas[i])
        if not np.isfinite(lp):
            out[i] = -np.inf
            continue
        ll = loglik_lognormal(thetas[i])
        out[i] = lp + ll
    return out

# --------------------------------------
# GMA utilities (anisotropic diagonal Gaussians)
# --------------------------------------
def diag_gauss_logpdf(X, mean, var_vec):
    # X: (NM,d), mean: (d,), var_vec: (d,)
    diff = X - mean
    inv = 1.0 / var_vec
    maha = np.sum(diff*diff * inv, axis=1)
    logdet = np.sum(np.log(var_vec))
    d = X.shape[1]
    return -0.5*(maha + logdet + d*np.log(2*np.pi))

def project_to_simplex(v):
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n+1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(v - theta, 0.0)

def rowwise_logsumexp(A):
    # A: (NM, N) => lse over axis=1
    m = np.max(A, axis=1)
    return m + np.log(np.sum(np.exp(A - m[:, None]), axis=1))

# --------------------------------------
# One GMA run given a bank (means, cov_diag, samples)
# Returns final weights, theta_samples, and weight history
# --------------------------------------
def run_gma(bank_samples, means, cov_diag, N, M, K, eta0):
    flat_samples = bank_samples.reshape(N*M, bank_samples.shape[-1])
    NM, d = flat_samples.shape

    # Precompute log-PDF matrix (NM x N)
    print("[GMA] Precomputing log-PDF matrix ...")
    logP = np.empty((NM, N), dtype=np.float64)
    for l in tqdm(range(N), desc="logP cols"):
        logP[:, l] = diag_gauss_logpdf(flat_samples, means[l], cov_diag)

    print("[GMA] Precomputing log target densities ...")
    log_p_target = log_unnormalized_p_thetas(flat_samples)

    # init weights
    w_hist = np.zeros((N, K+1), dtype=np.float64)
    w_hist[:, 0] = 1.0 / N

    for k in tqdm(range(1, K+1), desc="GMA (pGD)"):
        # stable log-mixture: log q(z_m) = logsumexp_l [ log w_l + logP[m,l] ]
        logw = np.log(np.maximum(w_hist[:, k-1], 1e-300))
        log_q = rowwise_logsumexp(logP + logw[None, :])

        # gradient per component i over its M-slice
        g = np.empty(N, dtype=np.float64)
        for i in range(N):
            s = i * M
            e = s + M
            g[i] = 1.0 + (log_q[s:e] - log_p_target[s:e]).mean()

        v = w_hist[:, k-1] - (eta0 / k) * g
        w_hist[:, k] = project_to_simplex(v)

    final_w = w_hist[:, -1]
    if not np.isfinite(final_w).all() or not np.isclose(final_w.sum(), 1.0):
        raise RuntimeError("Final weights invalid.")

    # ensemble sampling from fixed bank
    sel_comp = rng.choice(N, N*M, replace=True, p=final_w)
    sel_within = rng.integers(0, M, size=N*M)
    theta_samples = bank_samples[sel_comp, sel_within]
    return final_w, theta_samples, w_hist

# ----------------------------
# Hyperparameters & anisotropic covariances (tighter)
# ----------------------------
d = 8
N = 200
M = 30
K = 500
eta0 = 0.15

# Stage-1 stds on log scale (reduced 30–40%)
# [log a, log b, log c, log d, log X0, log Y0, log s1, log s2]
# Stage-1 stds on log scale (widen beta, delta a bit)
# [log a, log b, log c, log d, log X0, log Y0, log s1, log s2]
stds_stage1 = np.array([0.28, 0.22, 0.24, 0.28, 0.55, 0.55, 0.22, 0.22])
cov_diag_stage1 = stds_stage1**2

# centers near prior means
centers = np.array([
    0.0,                  # log a ~ log 1
    np.log(0.05),         # log b
    np.log(0.05),         # log delta
    0.0,                  # log gamma
    np.log(10.0),         # log X0
    np.log(10.0),         # log Y0
    -1.0,                 # log sigma1
    -1.0                  # log sigma2
], dtype=np.float64)

print(f"[Setup] d={d}, N={N}, M={M}, K={K}")

# ----------------------------
# Stage 1: coarse exploration
# ----------------------------
start = time.time()

means1 = rng.normal(loc=centers, scale=stds_stage1, size=(N, d))
bank_samples1 = rng.normal(size=(N, M, d)) * stds_stage1 + means1[:, None, :]
w1, theta_samples1, w_hist1 = run_gma(bank_samples1, means1, cov_diag_stage1, N, M, K, eta0)

# ----------------------------
# Stage 2: refine around top-weight components (tighter)
# ----------------------------
top_k = max(int(0.20*N), 1)                     # slightly fewer, more focused
top_idx = np.argsort(w1)[-top_k:]
top_w = w1[top_idx] / w1[top_idx].sum()
means_top = means1[top_idx]

refine_factor = 0.35                             # was 0.50
refine_stds  = stds_stage1 * refine_factor
jitter_scale = 0.15                              # was 0.30

means2 = means_top[rng.choice(top_k, N, replace=True, p=top_w)] \
         + rng.normal(size=(N, d)) * (refine_stds * jitter_scale)

cov_diag_stage2 = refine_stds**2
bank_samples2 = rng.normal(size=(N, M, d)) * refine_stds + means2[:, None, :]
w2, theta_samples2, w_hist2 = run_gma(bank_samples2, means2, cov_diag_stage2, N, M, K, eta0)

end = time.time()
print(f"[GMA] Total time (both stages): {end - start:.2f} s")

# Choose final samples from Stage 2
theta_samples = theta_samples2

# --------------------------------------
# Posterior summaries + predictive bands
# --------------------------------------
theta_mean = np.mean(theta_samples, axis=0)
a_m, b_m, c_m, d_m, X0_m, Y0_m, s1_m, s2_m = np.exp(theta_mean)

print("\n[Posterior mean (exp of log-params) — Stage 2]")
print(f"alpha={a_m:.4f}, beta={b_m:.5f}, delta={c_m:.5f}, gamma={d_m:.4f}, "
      f"X0={X0_m:.2f}, Y0={Y0_m:.2f}, sigma1={s1_m:.3f}, sigma2={s2_m:.3f}")

# Predictive bands
S_plot = min(2000, theta_samples.shape[0])
idx_plot = rng.choice(theta_samples.shape[0], S_plot, replace=False)
X_mat = np.empty((S_plot, len(t_obs)))
Y_mat = np.empty_like(X_mat)
for i, idx in enumerate(idx_plot):
    Xp, Yp = simulate_from_theta(theta_samples[idx])
    X_mat[i] = Xp
    Y_mat[i] = Yp

def bands(mat):
    return np.percentile(mat, [5, 50, 95], axis=0)

X_q = bands(X_mat)
Y_q = bands(Y_mat)

plt.figure(figsize=(11,5))
plt.subplot(1,2,1)
plt.title("Hare (prey)")
plt.plot(t_obs+years[0], hare, 'o', label="Obs")
plt.plot(t_obs+years[0], X_q[1], '-', label="Median")
plt.fill_between(t_obs+years[0], X_q[0], X_q[2], alpha=0.3, label="90% band")
plt.xlabel("Year"); plt.ylabel("Pelts (×10^3)"); plt.legend()

plt.subplot(1,2,2)
plt.title("Lynx (predator)")
plt.plot(t_obs+years[0], lynx, 'o', label="Obs")
plt.plot(t_obs+years[0], Y_q[1], '-', label="Median")
plt.fill_between(t_obs+years[0], Y_q[0], Y_q[2], alpha=0.3, label="90% band")
plt.xlabel("Year"); plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------
# Weight-evolution plots (top few components)
# --------------------------------------
def plot_weight_evolution(w_hist, title, top=12):
    Nw, Kp1 = w_hist.shape
    final = w_hist[:, -1]
    top_idx = np.argsort(final)[-top:][::-1]
    xs = np.arange(Kp1)
    plt.figure(figsize=(8,4.8))
    for i in top_idx:
        plt.plot(xs, np.maximum(w_hist[i], 1e-16), linewidth=1.2, label=f"comp {i}")
    plt.yscale('log')
    plt.xlabel("Iteration k"); plt.ylabel("Weight (log-scale)")
    plt.title(title)
    plt.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    plt.show()

plot_weight_evolution(w_hist1, "GMM weight evolution (Stage 1) — top 12")
plot_weight_evolution(w_hist2, "GMM weight evolution (Stage 2) — top 12")

# --------------------------------------
# Posterior histograms (natural scale) + bank overlays + reference lines
# --------------------------------------
nat_samples = np.exp(theta_samples)                     # final ensemble (posterior)
bank1_nat  = np.exp(bank_samples1.reshape(-1, d))      # Stage-1 bank
bank2_nat  = np.exp(bank_samples2.reshape(-1, d))      # Stage-2 bank

labels = [r"$\alpha$", r"$\beta$", r"$\delta$", r"$\gamma$",
          r"$X_0$", r"$Y_0$", r"$\sigma_1$", r"$\sigma_2$"]
ref_order = ["alpha","beta","delta","gamma","X0","Y0","sigma1","sigma2"]

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.ravel()
for i in range(8):
    post  = nat_samples[:, i]
    b1    = bank1_nat[:, i]
    b2    = bank2_nat[:, i]
    # common bin edges using combined central range
    combo = np.concatenate([post, b1, b2])
    lo = np.percentile(combo, 0.5); hi = np.percentile(combo, 99.5)
    edges = np.linspace(lo, hi, 60)

    axes[i].hist(post, bins=edges, density=True, alpha=0.65, label="Posterior")
    axes[i].hist(b1,   bins=edges, density=True, histtype="step", linewidth=2.0, label="Bank (stage 1)")
    axes[i].hist(b2,   bins=edges, density=True, histtype="step", linewidth=2.0, linestyle="--", label="Bank (stage 2)")
    axes[i].set_title(labels[i])

    # ground-truth vertical line
    rv = ref_vals.get(ref_order[i], None)
    if rv is not None and np.isfinite(rv):
        axes[i].axvline(rv, color="red", linewidth=2.0, label="Ref" if i==0 else None)

    if i == 0:
        axes[i].legend(loc="best", fontsize=9)

plt.tight_layout()
plt.show()

# --------------------------------------
# 1x4 2-D scatter panel: (alpha,beta), (delta,gamma), (sigma1,sigma2), (X0,Y0)
# --------------------------------------
def subsample_rows(A, max_n=4000):
    n = A.shape[0]
    if n <= max_n:
        return A
    return A[rng.choice(n, max_n, replace=False)]

B1   = subsample_rows(bank1_nat, 4000)
B2   = subsample_rows(bank2_nat, 4000)
POST = subsample_rows(nat_samples, 4000)

pairs = [(0,1), (2,3), (6,7), (4,5)]
pair_labels = [(r"$\alpha$", r"$\beta$"),
               (r"$\delta$", r"$\gamma$"),
               (r"$\sigma_1$", r"$\sigma_2$"),
               (r"$X_0$", r"$Y_0$")]
ref_points = [
    (ref_vals["alpha"], ref_vals["beta"]),
    (ref_vals["delta"], ref_vals["gamma"]),
    (ref_vals["sigma1"], ref_vals["sigma2"]),
    (ref_vals["X0"],    ref_vals["Y0"]),
]

plt.figure(figsize=(16,4))
for k,(i,j) in enumerate(pairs, start=1):
    plt.subplot(1,4,k)
    plt.scatter(B1[:,i], B1[:,j], s=6, alpha=0.18, label="Stage-1 bank")
    plt.scatter(B2[:,i], B2[:,j], s=6, alpha=0.18, label="Stage-2 bank")
    plt.scatter(POST[:,i], POST[:,j], s=10, alpha=0.5, label="Posterior")
    rx, ry = ref_points[k-1]
    plt.plot([rx],[ry], marker="*", markersize=10, color="red", label="Ref" if k==1 else None)
    plt.xlabel(pair_labels[k-1][0]); plt.ylabel(pair_labels[k-1][1])
    if k==1:
        plt.legend(loc="best")
plt.tight_layout()
plt.show()

rmse_hare, rmse_lynx = np.sqrt(np.mean((X_q[1]-hare)**2)), np.sqrt(np.mean((Y_q[1]-lynx)**2))
print(f"[Posterior] RMSE hare={rmse_hare:.2f}, lynx={rmse_lynx:.2f}")

"""# MD-GMA."""

# =========================================================
# LV + GMA (refined, minimal changes) + diagnostics
#   - RK4 (dt=0.005)
#   - Log-sum-exp mixture evaluation with precomputed log-PDFs
#   - Anisotropic diagonal covariances per dimension (tighter)
#   - 2-stage refine (coarse -> focused around top-weight comps)
#   - Two separate sigmas (hare, lynx), Stan-matching priors
#   - NEW: ground-truth lines; hist overlays from bank1 & bank2;
#          1x4 2-D scatter panel; weight-evolution plots
# =========================================================
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from math import log, pi
import time

rng = np.random.default_rng(123)

# ----------------------------
# Data
# ----------------------------
years = np.array([1900,1901,1902,1903,1904,1905,1906,1907,1908,1909,
                  1910,1911,1912,1913,1914,1915,1916,1917,1918,1919,1920], dtype=np.int64)
lynx = np.array([4.0, 6.1, 9.8, 35.2, 59.4, 41.7, 19.0, 13.0, 8.3, 9.1,
                 7.4, 8.0, 12.3, 19.5, 45.7, 51.1, 29.7, 15.8, 9.7, 10.1, 8.6], dtype=np.float64)
hare = np.array([30.0, 47.2, 70.2, 77.4, 36.3, 20.6, 18.1, 21.4, 22.0, 25.4,
                 27.1, 40.3, 57.0, 76.6, 52.3, 19.5, 11.2, 7.6, 14.6, 16.2, 24.7], dtype=np.float64)

t_obs = (years - years[0]).astype(float)  # 0..20
dt_rk = 0.005                             # smaller RK4 step for fidelity

# Reference (ground-truth) values used for vertical lines
ref_vals = {
    "alpha": 0.55,
    "beta": 0.028,
    "delta": 0.024,
    "gamma": 0.80,
    "X0": 33.956,
    "Y0": 5.933,
    "sigma1": 0.25,
    "sigma2": 0.25,
}

# --------------------------------------
# LV dynamics + RK4 integrator
# --------------------------------------
def lv_rhs(state, p):
    X, Y = state
    a, b, c, d = p
    return np.array([a*X - b*X*Y, c*X*Y - d*Y], dtype=np.float64)

def rk4_integrate(p, X0, Y0, t_eval, dt=dt_rk):
    assert t_eval[0] == 0.0
    steps_total = int(np.ceil((t_eval[-1] - 0.0) / dt))
    state = np.array([X0, Y0], dtype=np.float64)
    t_curr = 0.0
    out = np.empty((len(t_eval), 2), dtype=np.float64)
    next_idx = 0
    out[next_idx] = state
    next_idx += 1
    for _ in range(steps_total):
        if next_idx >= len(t_eval):
            break
        k1 = lv_rhs(state, p)
        k2 = lv_rhs(state + 0.5*dt*k1, p)
        k3 = lv_rhs(state + 0.5*dt*k2, p)
        k4 = lv_rhs(state + dt*k3, p)
        state = state + (dt/6.0)*(k1 + 2*k2 + 2*k3 + k4)
        state = np.maximum(state, 1e-12)  # keep positive
        t_curr += dt
        while next_idx < len(t_eval) and t_curr + 1e-12 >= t_eval[next_idx]:
            out[next_idx] = state
            next_idx += 1
    return out

# --------------------------------------
# Parameterization / priors / likelihood
# theta = [log a, log b, log c, log d, log X0, log Y0, log sigma1, log sigma2]
# --------------------------------------
def unpack_theta(theta):
    la, lb, lc, ld, lX0, lY0, ls1, ls2 = theta
    a, b, c, d = np.exp([la, lb, lc, ld])
    X0, Y0 = np.exp([lX0, lY0])
    s1, s2 = np.exp([ls1, ls2])
    return a, b, c, d, X0, Y0, s1, s2

def simulate_from_theta(theta):
    a, b, c, d, X0, Y0, _, _ = unpack_theta(theta)
    traj = rk4_integrate([a, b, c, d], X0, Y0, t_obs)
    return traj[:, 0], traj[:, 1]

def log_normal_pdf(x, mu, sd):
    z = (x - mu) / sd
    return -0.5*z*z - log(sd) - 0.5*log(2*pi)

def log_prior(theta):
    la, lb, lc, ld, lX0, lY0, ls1, ls2 = theta
    a, b, c, d = np.exp([la, lb, lc, ld])
    if not np.isfinite(a*b*c*d) or (a<=0 or b<=0 or c<=0 or d<=0):
        return -np.inf
    lp = 0.0
    # natural-scale Normals (+ Jacobian terms)
    lp += log_normal_pdf(a, 1.0, 0.5) + la       # alpha
    lp += log_normal_pdf(d, 1.0, 0.5) + ld       # gamma
    lp += log_normal_pdf(b, 0.05, 0.05) + lb     # beta
    lp += log_normal_pdf(c, 0.05, 0.05) + lc     # delta
    # LogNormals (Normal on logs)
    mu_lN = np.log(10.0)
    lp += log_normal_pdf(lX0, mu_lN, 1.0)        # log X0
    lp += log_normal_pdf(lY0, mu_lN, 1.0)        # log Y0
    lp += log_normal_pdf(ls1, -1.0, 1.0)         # log sigma1
    lp += log_normal_pdf(ls2, -1.0, 1.0)         # log sigma2
    return lp

def loglik_lognormal(theta):
    a, b, c, d, X0, Y0, s1, s2 = unpack_theta(theta)
    if (s1 <= 1e-10) or (s2 <= 1e-10) or (not np.isfinite(s1)) or (not np.isfinite(s2)):
        return -np.inf
    Xp, Yp = simulate_from_theta(theta)
    Xp = np.maximum(Xp, 1e-12)
    Yp = np.maximum(Yp, 1e-12)
    rx = np.log(hare) - np.log(Xp)
    ry = np.log(lynx) - np.log(Yp)
    nx, ny = rx.size, ry.size
    return (
        -0.5*np.sum((rx*rx)/(s1*s1)) - nx*np.log(s1) - 0.5*nx*np.log(2*np.pi)
        -0.5*np.sum((ry*ry)/(s2*s2)) - ny*np.log(s2) - 0.5*ny*np.log(2*np.pi)
    )

def log_unnormalized_p_thetas(thetas):
    out = np.empty(thetas.shape[0], dtype=np.float64)
    for i in range(thetas.shape[0]):
        lp = log_prior(thetas[i])
        if not np.isfinite(lp):
            out[i] = -np.inf
            continue
        ll = loglik_lognormal(thetas[i])
        out[i] = lp + ll
    return out

# --------------------------------------
# GMA utilities (anisotropic diagonal Gaussians)
# --------------------------------------
def diag_gauss_logpdf(X, mean, var_vec):
    # X: (NM,d), mean: (d,), var_vec: (d,)
    diff = X - mean
    inv = 1.0 / var_vec
    maha = np.sum(diff*diff * inv, axis=1)
    logdet = np.sum(np.log(var_vec))
    d = X.shape[1]
    return -0.5*(maha + logdet + d*np.log(2*np.pi))

def project_to_simplex(v):
    n = v.shape[0]
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u) - 1
    ind = np.arange(1, n+1)
    cond = u - cssv / ind > 0
    rho = ind[cond][-1]
    theta = cssv[cond][-1] / rho
    return np.maximum(v - theta, 0.0)

def rowwise_logsumexp(A):
    # A: (NM, N) => lse over axis=1
    m = np.max(A, axis=1)
    return m + np.log(np.sum(np.exp(A - m[:, None]), axis=1))

# --------------------------------------
# One GMA run given a bank (means, cov_diag, samples)
# Returns final weights, theta_samples, and weight history
# --------------------------------------
def run_gma(bank_samples, means, cov_diag, N, M, K, eta0):
    flat_samples = bank_samples.reshape(N*M, bank_samples.shape[-1])
    NM, d = flat_samples.shape

    # Precompute log-PDF matrix (NM x N)
    print("[GMA] Precomputing log-PDF matrix ...")
    logP = np.empty((NM, N), dtype=np.float64)
    for l in tqdm(range(N), desc="logP cols"):
        logP[:, l] = diag_gauss_logpdf(flat_samples, means[l], cov_diag)

    print("[GMA] Precomputing log target densities ...")
    log_p_target = log_unnormalized_p_thetas(flat_samples)

    # init weights
    w_hist = np.zeros((N, K+1), dtype=np.float64)
    w_hist[:, 0] = 1.0 / N

    for k in tqdm(range(1, K+1), desc="GMA (MD)"):
        # stable log-mixture: log q(z_m) = logsumexp_l [ log w_l + logP[m,l] ]
        logw = np.log(np.maximum(w_hist[:, k-1], 1e-300))
        log_q = rowwise_logsumexp(logP + logw[None, :])

        # gradient per component i over its M-slice
        g = np.empty(N, dtype=np.float64)
        for i in range(N):
            s = i * M
            e = s + M
            g[i] = 1.0 + (log_q[s:e] - log_p_target[s:e]).mean()

        # ---- Mirror Descent (Exponentiated Gradient) update ----
        eta = eta0 / k
        w_prev = w_hist[:, k-1]
        w_tilde = w_prev * np.exp(-eta * g)
        w_hist[:, k] = w_tilde / np.sum(w_tilde)
        # --------------------------------------------------------

    final_w = w_hist[:, -1]
    if not np.isfinite(final_w).all() or not np.isclose(final_w.sum(), 1.0):
        raise RuntimeError("Final weights invalid.")

    # ensemble sampling from fixed bank
    sel_comp = rng.choice(N, N*M, replace=True, p=final_w)
    sel_within = rng.integers(0, M, size=N*M)
    theta_samples = bank_samples[sel_comp, sel_within]
    return final_w, theta_samples, w_hist

# ----------------------------
# Hyperparameters & anisotropic covariances (tighter)
# ----------------------------
d = 8
N = 200
M = 30
K = 500
eta0 = 0.15

# Stage-1 stds on log scale (reduced 30–40%)
# [log a, log b, log c, log d, log X0, log Y0, log s1, log s2]
# Stage-1 stds on log scale (widen beta, delta a bit)
# [log a, log b, log c, log d, log X0, log Y0, log s1, log s2]
stds_stage1 = np.array([0.28, 0.22, 0.24, 0.28, 0.55, 0.55, 0.22, 0.22])
cov_diag_stage1 = stds_stage1**2

# centers near prior means
centers = np.array([
    0.0,                  # log a ~ log 1
    np.log(0.05),         # log b
    np.log(0.05),         # log delta
    0.0,                  # log gamma
    np.log(10.0),         # log X0
    np.log(10.0),         # log Y0
    -1.0,                 # log sigma1
    -1.0                  # log sigma2
], dtype=np.float64)

print(f"[Setup] d={d}, N={N}, M={M}, K={K}")

# ----------------------------
# Stage 1: coarse exploration
# ----------------------------
start = time.time()

means1 = rng.normal(loc=centers, scale=stds_stage1, size=(N, d))
bank_samples1 = rng.normal(size=(N, M, d)) * stds_stage1 + means1[:, None, :]
w1, theta_samples1, w_hist1 = run_gma(bank_samples1, means1, cov_diag_stage1, N, M, K, eta0)

# ----------------------------
# Stage 2: refine around top-weight components (tighter)
# ----------------------------
top_k = max(int(0.20*N), 1)                     # slightly fewer, more focused
top_idx = np.argsort(w1)[-top_k:]
top_w = w1[top_idx] / w1[top_idx].sum()
means_top = means1[top_idx]

refine_factor = 0.35                             # was 0.50
refine_stds  = stds_stage1 * refine_factor
jitter_scale = 0.15                              # was 0.30

means2 = means_top[rng.choice(top_k, N, replace=True, p=top_w)] \
         + rng.normal(size=(N, d)) * (refine_stds * jitter_scale)

cov_diag_stage2 = refine_stds**2
bank_samples2 = rng.normal(size=(N, M, d)) * refine_stds + means2[:, None, :]
w2, theta_samples2, w_hist2 = run_gma(bank_samples2, means2, cov_diag_stage2, N, M, K, eta0)

end = time.time()
print(f"[GMA] Total time (both stages): {end - start:.2f} s")

# Choose final samples from Stage 2
theta_samples = theta_samples2

# --------------------------------------
# Posterior summaries + predictive bands
# --------------------------------------
theta_mean = np.mean(theta_samples, axis=0)
a_m, b_m, c_m, d_m, X0_m, Y0_m, s1_m, s2_m = np.exp(theta_mean)

print("\n[Posterior mean (exp of log-params) — Stage 2]")
print(f"alpha={a_m:.4f}, beta={b_m:.5f}, delta={c_m:.5f}, gamma={d_m:.4f}, "
      f"X0={X0_m:.2f}, Y0={Y0_m:.2f}, sigma1={s1_m:.3f}, sigma2={s2_m:.3f}")

# Predictive bands
S_plot = min(2000, theta_samples.shape[0])
idx_plot = rng.choice(theta_samples.shape[0], S_plot, replace=False)
X_mat = np.empty((S_plot, len(t_obs)))
Y_mat = np.empty_like(X_mat)
for i, idx in enumerate(idx_plot):
    Xp, Yp = simulate_from_theta(theta_samples[idx])
    X_mat[i] = Xp
    Y_mat[i] = Yp

def bands(mat):
    return np.percentile(mat, [5, 50, 95], axis=0)

X_q = bands(X_mat)
Y_q = bands(Y_mat)

plt.figure(figsize=(11,5))
plt.subplot(1,2,1)
plt.title("Hare (prey)")
plt.plot(t_obs+years[0], hare, 'o', label="Obs")
plt.plot(t_obs+years[0], X_q[1], '-', label="Median")
plt.fill_between(t_obs+years[0], X_q[0], X_q[2], alpha=0.3, label="90% band")
plt.xlabel("Year"); plt.ylabel("Pelts (×10^3)"); plt.legend()

plt.subplot(1,2,2)
plt.title("Lynx (predator)")
plt.plot(t_obs+years[0], lynx, 'o', label="Obs")
plt.plot(t_obs+years[0], Y_q[1], '-', label="Median")
plt.fill_between(t_obs+years[0], Y_q[0], Y_q[2], alpha=0.3, label="90% band")
plt.xlabel("Year"); plt.legend()
plt.tight_layout()
plt.show()

# --------------------------------------
# Weight-evolution plots (top few components)
# --------------------------------------
def plot_weight_evolution(w_hist, title, top=12):
    Nw, Kp1 = w_hist.shape
    final = w_hist[:, -1]
    top_idx = np.argsort(final)[-top:][::-1]
    xs = np.arange(Kp1)
    plt.figure(figsize=(8,4.8))
    for i in top_idx:
        plt.plot(xs, np.maximum(w_hist[i], 1e-16), linewidth=1.2, label=f"comp {i}")
    plt.yscale('log')
    plt.xlabel("Iteration k"); plt.ylabel("Weight (log-scale)")
    plt.title(title)
    plt.legend(ncol=6, fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.12))
    plt.tight_layout()
    plt.show()

plot_weight_evolution(w_hist1, "GMM weight evolution (Stage 1) — top 12")
plot_weight_evolution(w_hist2, "GMM weight evolution (Stage 2) — top 12")

# --------------------------------------
# Posterior histograms (natural scale) + bank overlays + reference lines
# --------------------------------------
nat_samples = np.exp(theta_samples)                     # final ensemble (posterior)
bank1_nat  = np.exp(bank_samples1.reshape(-1, d))      # Stage-1 bank
bank2_nat  = np.exp(bank_samples2.reshape(-1, d))      # Stage-2 bank

labels = [r"$\alpha$", r"$\beta$", r"$\delta$", r"$\gamma$",
          r"$X_0$", r"$Y_0$", r"$\sigma_1$", r"$\sigma_2$"]
ref_order = ["alpha","beta","delta","gamma","X0","Y0","sigma1","sigma2"]

fig, axes = plt.subplots(2, 4, figsize=(14, 7))
axes = axes.ravel()
for i in range(8):
    post  = nat_samples[:, i]
    b1    = bank1_nat[:, i]
    b2    = bank2_nat[:, i]
    # common bin edges using combined central range
    combo = np.concatenate([post, b1, b2])
    lo = np.percentile(combo, 0.5); hi = np.percentile(combo, 99.5)
    edges = np.linspace(lo, hi, 60)

    axes[i].hist(post, bins=edges, density=True, alpha=0.65, label="Posterior")
    axes[i].hist(b1,   bins=edges, density=True, histtype="step", linewidth=2.0, label="Bank (stage 1)")
    axes[i].hist(b2,   bins=edges, density=True, histtype="step", linewidth=2.0, linestyle="--", label="Bank (stage 2)")
    axes[i].set_title(labels[i])

    # ground-truth vertical line
    rv = ref_vals.get(ref_order[i], None)
    if rv is not None and np.isfinite(rv):
        axes[i].axvline(rv, color="red", linewidth=2.0, label="Ref" if i==0 else None)

    if i == 0:
        axes[i].legend(loc="best", fontsize=9)

plt.tight_layout()
plt.show()

# --------------------------------------
# 1x4 2-D scatter panel: (alpha,beta), (delta,gamma), (sigma1,sigma2), (X0,Y0)
# --------------------------------------
def subsample_rows(A, max_n=4000):
    n = A.shape[0]
    if n <= max_n:
        return A
    return A[rng.choice(n, max_n, replace=False)]

B1   = subsample_rows(bank1_nat, 4000)
B2   = subsample_rows(bank2_nat, 4000)
POST = subsample_rows(nat_samples, 4000)

pairs = [(0,1), (2,3), (6,7), (4,5)]
pair_labels = [(r"$\alpha$", r"$\beta$"),
               (r"$\delta$", r"$\gamma$"),
               (r"$\sigma_1$", r"$\sigma_2$"),
               (r"$X_0$", r"$Y_0$")]
ref_points = [
    (ref_vals["alpha"], ref_vals["beta"]),
    (ref_vals["delta"], ref_vals["gamma"]),
    (ref_vals["sigma1"], ref_vals["sigma2"]),
    (ref_vals["X0"],    ref_vals["Y0"]),
]

plt.figure(figsize=(16,4))
for k,(i,j) in enumerate(pairs, start=1):
    plt.subplot(1,4,k)
    plt.scatter(B1[:,i], B1[:,j], s=6, alpha=0.18, label="Stage-1 bank")
    plt.scatter(B2[:,i], B2[:,j], s=6, alpha=0.18, label="Stage-2 bank")
    plt.scatter(POST[:,i], POST[:,j], s=10, alpha=0.5, label="Posterior")
    rx, ry = ref_points[k-1]
    plt.plot([rx],[ry], marker="*", markersize=10, color="red", label="Ref" if k==1 else None)
    plt.xlabel(pair_labels[k-1][0]); plt.ylabel(pair_labels[k-1][1])
    if k==1:
        plt.legend(loc="best")
plt.tight_layout()
plt.show()

rmse_hare, rmse_lynx = np.sqrt(np.mean((X_q[1]-hare)**2)), np.sqrt(np.mean((Y_q[1]-lynx)**2))
print(f"[Posterior] RMSE hare={rmse_hare:.2f}, lynx={rmse_lynx:.2f}")

"""# end."""
