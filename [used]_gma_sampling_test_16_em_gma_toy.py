# -*- coding: utf-8 -*-
"""[used] GMA sampling test 16: EM-GMA toy.ipynb

#yongchao.huang@abdn.ac.uk
"""

!lscpu

import psutil

print("CPU cores:", psutil.cpu_count(logical=False))
print("Logical CPUs:", psutil.cpu_count(logical=True))
print("Memory (GB):", round(psutil.virtual_memory().total / 1e9, 2))
print("Disk space (GB):", round(psutil.disk_usage('/').total / 1e9, 2))

"""# Begin.

# ground truth: GMM.
"""

# em_gma_demo.py
# ------------------------------------------------------------
# Ground-truth GMM sampling + (A) standard EM-on-data (MLE)
#                      and   (B) population EM (EM-GMA, SNIS)
# ------------------------------------------------------------
import numpy as np
from numpy.linalg import cholesky, eigh

rng = np.random.default_rng(42)

# ---------- utilities ----------
def logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    out = m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis)

def make_spd(S, eps=1e-6):
    # project symmetric matrix to SPD by eigenvalue floor
    evals, evecs = eigh(0.5*(S+S.T))
    evals = np.clip(evals, eps, None)
    return (evecs * evals) @ evecs.T

def mvn_logpdf(x, mu, Sigma, L=None):
    """
    x: (M,d), mu: (d,), Sigma: (d,d). If L (chol) given, reuse it.
    """
    d = mu.shape[0]
    if L is None:
        L = cholesky(Sigma)
    diff = x - mu
    # solve L y = diff^T => y = L^{-1} diff^T
    sol = np.linalg.solve(L, diff.T)
    quad = np.sum(sol**2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (quad + d*np.log(2*np.pi) + logdet)

def gmm_logpdf(x, w, mus, Sigmas, chols=None):
    """
    x: (M,d), w: (K,), mus: (K,d), Sigmas: (K,d,d)
    """
    K = len(w)
    if chols is None:
        chols = [cholesky(Sigmas[k]) for k in range(K)]
    parts = np.stack([np.log(w[k]) + mvn_logpdf(x, mus[k], Sigmas[k], chols[k]) for k in range(K)], axis=1)  # (M,K)
    return logsumexp(parts, axis=1), parts, chols

def sample_gmm(n, w, mus, Sigmas):
    K, d = mus.shape
    comps = rng.choice(K, size=n, p=w)
    x = np.empty((n, d))
    for k in range(K):
        idx = np.where(comps == k)[0]
        if idx.size:
            L = cholesky(Sigmas[k])
            eps = rng.normal(size=(idx.size, d))
            x[idx] = mus[k] + eps @ L.T
    return x

def init_from_data(X, K, ridge=1e-3):
    """Simple init: random means from data; shared covariance; uniform weights."""
    n, d = X.shape
    mus = X[rng.choice(n, size=K, replace=False)].copy()
    Xc = X - X.mean(0)
    S = (Xc.T @ Xc) / n + ridge * np.eye(d)
    Sigmas = np.stack([S.copy() for _ in range(K)], axis=0)
    w = np.ones(K) / K
    return w, mus, Sigmas

# ---------- (A) standard EM on observed data ----------
def em_on_data(X, K, n_iter=50, ridge=1e-6):
    n, d = X.shape
    w, mus, Sigmas = init_from_data(X, K, ridge=ridge)
    for t in range(n_iter):
        # E-step
        log_qx, parts, chols = gmm_logpdf(X, w, mus, Sigmas)
        # responsibilities r_{nk} = w_k N_k(x_n)/q(x_n)
        R = np.exp(parts - log_qx[:, None])  # (n,K)
        Nk = R.sum(axis=0) + 1e-16
        # M-step
        w = Nk / n
        mus = (R.T @ X) / Nk[:, None]
        for k in range(K):
            Xc = X - mus[k]
            Sk = (R[:, k][:, None] * Xc).T @ Xc / Nk[k]
            Sigmas[k] = make_spd(Sk + ridge * np.eye(d))
    return w, mus, Sigmas

# ---------- (B) population EM (EM-GMA) with SNIS ----------
def em_gma_population(
    bar_p_logpdf,           # function: (M,d)->(M,) unnormalised log target
    K, d,                   # mixture size, dim
    M_bank=4096,            # bank size per sweep
    n_iter=50,              # EM sweeps
    ridge=1e-6,
    init="data",            # "data" or "random"
    X_data=None,            # if init=="data", seed from data
    w0=None, mus0=None, Sigmas0=None
):
    # init params
    if (w0 is not None) and (mus0 is not None) and (Sigmas0 is not None):
        w, mus, Sigmas = w0.copy(), mus0.copy(), Sigmas0.copy()
    elif init == "data" and X_data is not None:
        w, mus, Sigmas = init_from_data(X_data, K, ridge=ridge)
    else:
        w = np.ones(K) / K
        mus = rng.normal(0, 1.0, size=(K, d))
        Sigmas = np.stack([np.eye(d) for _ in range(K)], axis=0)

    for t in range(n_iter):
        # --- bank: draw from current proposal r^{(t)} = q_theta^{(t)}
        Z = sample_gmm(M_bank, w, mus, Sigmas)  # (M_bank, d)
        # log densities
        log_qz, parts, chols = gmm_logpdf(Z, w, mus, Sigmas)
        log_pbar = bar_p_logpdf(Z)             # unnormalised log target
        # SNIS weights
        lw = log_pbar - log_qz
        lw -= lw.max()                         # stabilise
        w_tilde = np.exp(lw)
        omega = w_tilde / (w_tilde.sum() + 1e-16)  # (M_bank,)

        # responsibilities under current q
        R = np.exp(parts - log_qz[:, None])  # (M_bank,K)
        # effective counts (p-weighted)
        Nk = (omega[:, None] * R).sum(axis=0) + 1e-16  # (K,)
        w = Nk / Nk.sum()

        # update means
        mus = ((omega[:, None] * R).T @ Z) / Nk[:, None]
        # update covariances
        for k in range(K):
            Zc = Z - mus[k]
            Sk = (omega[:, None] * R[:, [k]] * Zc).T @ Zc / Nk[k]
            Sigmas[k] = make_spd(Sk + ridge * np.eye(d))

    return w, mus, Sigmas

# ---------- main ----------
# Ground truth GMM in 2D
d = 2
K_true = 3
w_true = np.array([0.45, 0.25, 0.30])
mus_true = np.array([[0.0, 0.0],
                      [3.0, 1.5],
                      [-2.0, 3.0]])
Sigmas_true = np.stack([
    np.array([[1.0, 0.3],[0.3, 0.8]]),
    np.array([[0.5, 0.0],[0.0, 0.5]]),
    np.array([[0.8, -0.2],[-0.2, 0.6]])
], axis=0)
# normalise SPD just in case
Sigmas_true = np.stack([make_spd(S) for S in Sigmas_true], axis=0)

# Helper: ground-truth (normalised) log pdf as "unnormalised" target
def log_pbar(Z):
    lp, _, _ = gmm_logpdf(Z, w_true, mus_true, Sigmas_true)
    return lp  # already normalised, OK for SNIS

# 1) generate observed data
n_data = 5000
X = sample_gmm(n_data, w_true, mus_true, Sigmas_true)

# 2A) fit by standard EM on data
K_fit = 3
w_mle, mu_mle, Sig_mle = em_on_data(X, K_fit, n_iter=40, ridge=1e-6)

# 2B) fit by population EM (EM-GMA) against target density (no data needed)
w_gma, mu_gma, Sig_gma = em_gma_population(
    bar_p_logpdf=log_pbar, K=K_fit, d=d,
    M_bank=4096, n_iter=40, ridge=1e-6, init="random", X_data=None
)

# Report (sorted by mean x to mitigate label switching in the printout)
def sort_by_x(w, mus, Sigmas):
    order = np.argsort(mus[:, 0])
    return w[order], mus[order], Sigmas[order]

print("\n=== Ground truth ===")
wt, mt, St = sort_by_x(w_true, mus_true, Sigmas_true)
for k in range(K_true):
    print(f"k={k}: w={wt[k]:.3f}, mu={mt[k]}, diag(Sigma)={np.diag(St[k])}")

print("\n=== Standard EM (MLE on data) ===")
w1, m1, S1 = sort_by_x(w_mle, mu_mle, Sig_mle)
for k in range(K_fit):
    print(f"k={k}: w={w1[k]:.3f}, mu={m1[k]}, diag(Sigma)={np.diag(S1[k])}")

print("\n=== Population EM (EM-GMA, SNIS) ===")
w2, m2, S2 = sort_by_x(w_gma, mu_gma, Sig_gma)
for k in range(K_fit):
    print(f"k={k}: w={w2[k]:.3f}, mu={m2[k]}, diag(Sigma)={np.diag(S2[k])}")

# ---------- plotting ----------
import matplotlib.pyplot as plt

def ellipse_points(mu, Sigma, nsig=2.0, num=200):
    """Return points of the nsig-sigma covariance ellipse."""
    vals, vecs = np.linalg.eigh(Sigma)
    rad = np.sqrt(np.clip(vals, 0, None)) * nsig
    angles = np.linspace(0, 2*np.pi, num)
    circle = np.stack([np.cos(angles), np.sin(angles)], axis=0)  # (2,num)
    A = vecs @ np.diag(rad)  # (2,2)
    pts = (A @ circle).T + mu  # (num,2)
    return pts

def plot_gmm(ax, w, mus, Sigmas, color, label_prefix, nsig=2.0):
    K = len(w)
    # ellipses
    first = True
    for k in range(K):
        pts = ellipse_points(mus[k], Sigmas[k], nsig=nsig)
        ax.plot(pts[:,0], pts[:,1], color=color, label=(label_prefix if first else None))
        first = False
    # means
    ax.plot(mus[:,0], mus[:,1], "o", markerfacecolor=color, label=None)

# one chart, overlay everything
fig, ax = plt.subplots()

# optional: light scatter of data for context
sub = rng.choice(len(X), size=min(1000, len(X)), replace=False)
ax.scatter(X[sub,0], X[sub,1], s=4, alpha=0.2, label="data (subset)")

# overlays (sorted for stable legend if you used sort_by_x)
wt, mt, St = sort_by_x(w_true, mus_true, Sigmas_true)
w1, m1, S1   = sort_by_x(w_mle,  mu_mle,  Sig_mle)
w2, m2, S2   = sort_by_x(w_gma,  mu_gma,  Sig_gma)

plot_gmm(ax, wt, mt, St, color='black', label_prefix="ground truth", nsig=2.0)
plot_gmm(ax, w1, m1, S1, color='green',  label_prefix="EM on data",  nsig=2.0)
plot_gmm(ax, w2, m2, S2, color='red',  label_prefix="EM-GMA (population)", nsig=2.0)

ax.set_aspect("equal", adjustable="box")
ax.set_title("GMM: ground truth vs EM-on-data vs EM-GMA")
ax.legend()
ax.set_xlabel("x1")
ax.set_ylabel("x2")
plt.show()

"""# end."""
