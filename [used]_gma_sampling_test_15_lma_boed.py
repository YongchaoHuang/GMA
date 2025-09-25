# -*- coding: utf-8 -*-
"""[used] GMA sampling test 15: LMA-BOED.ipynb

#yongchao.huang@abdn.ac.uk
"""

!lscpu

import psutil

print("CPU cores:", psutil.cpu_count(logical=False))
print("Logical CPUs:", psutil.cpu_count(logical=True))
print("Memory (GB):", round(psutil.virtual_memory().total / 1e9, 2))
print("Disk space (GB):", round(psutil.disk_usage('/').total / 1e9, 2))

"""# Begin.

# LMA.
"""

# -*- coding: utf-8 -*-
"""
Bayesian OED for logistic dose–response via LMA (Laplace Mixture Approximation)

Includes:
  (1) Per-dose diagnostics: print pbar_k, Eh_k, Delta
  (2) Entropy-regularized greedy design (tau >= 0)
  (+) Baselines: Uniform and Fisher (local D-opt) allocations
  (+) Matplotlib plots: EDA (curves/summary/response), Delta_k, gamma (all designs), posterior contours
  (+) Common Random Numbers (CRNs) for fair design comparisons
  (+) Save figures to PNGs

Notes:
  - Dose predictor is centered (shared offset) for numerical stability.
  - Prior over θ=(α, β) built empirically from historical cell lines.
  - LMA prior surrogate via clustered local Gaussians; reverse-KL to fit ω.
  - EIG estimator uses LMA samples (no inner outcome sampling).
"""

import os, math, numpy as np, pandas as pd
from dataclasses import dataclass
from typing import Tuple, Dict

import matplotlib.pyplot as plt

# -----------------------------
# Config
# -----------------------------
@dataclass
class Config:
    CSV_PATH: str = "gdsc_like.csv"
    DRUG_NAME: str = "DrugA"
    TARGET_CELL: str = "CL_0001"
    BIN_THRESH: float = 0.5
    EPS_DOSE: float = 1e-8
    # Design / EIG
    S: int = 2000                    # LMA samples for EIG terms
    BUDGET_N: int = 21               # total replicates
    MIN_PER_DOSE: int = 1            # floor to avoid degenerate all-in-one-dose (set 0 to disable)
    ENTROPY_TAU: float = 0.02        # tau >= 0 for entropy-regularized greedy (softly spreads γ)
    RANDOM_SEED: int = 42
    # Logistic fits (centered predictor)
    L2_PREC: float = 1e-2
    MAX_ITERS: int = 100
    TOL: float = 1e-8
    # LMA / clustering
    J: int = 5
    RIDGE_COV: float = 1e-2
    KM_SEED: int = 123
    # Reverse-KL
    REVERSE_KL_ITERS: int = 500
    REVERSE_KL_STEP: float = 0.2
    # Posterior eval (Gaussian prior from LMA moment match)
    L2_DAMP: float = 1e-8
    # Synthetic fallback
    SYNTH_NUM_CELLS: int = 60
    SYNTH_DOSE_GRID = None
    SYNTH_THETA_MIX = None
    VERBOSE: bool = True

cfg = Config()

def print_example_data_table(df: pd.DataFrame, cfg, n_cells: int = 3, save_csv: bool = True):
    """
    Print a compact example table of the data used for the chosen drug.
    Includes the binarized response y (viability <= BIN_THRESH ? 1 : 0).
    """
    df_drug = df[df["drug"] == cfg.DRUG_NAME].copy()
    if df_drug.empty:
        print(f"[WARN] No rows for drug '{cfg.DRUG_NAME}'")
        return

    # Add binary response column consistent with the modeling code
    df_drug["y"] = (df_drug["viability"] <= cfg.BIN_THRESH).astype(int)

    # Pick target cell + a couple of others (deterministic order)
    cells = [cfg.TARGET_CELL] + [c for c in df_drug["cell_line"].unique() if c != cfg.TARGET_CELL]
    cells = cells[:n_cells]

    # Take one full dose series per chosen cell (7 doses each in our setup)
    tbl = (
        df_drug[df_drug["cell_line"].isin(cells)]
        .sort_values(["cell_line", "dose"])
        .loc[:, ["cell_line", "dose", "viability", "y"]]
    )

    # Pretty print (first 7*n_cells rows is one curve per chosen cell)
    rows_to_show = min(len(tbl), 7 * len(cells))
    fmt = {
        "dose": lambda v: f"{v:.3g}",        # scientific-ish
        "viability": lambda v: f"{v:.3f}",   # 3 dp
    }
    print("\n=== Example data (drug={}, target={}, thresh={}) ==="
          .format(cfg.DRUG_NAME, cfg.TARGET_CELL, cfg.BIN_THRESH))
    print(tbl.head(rows_to_show).to_string(index=False, formatters=fmt))

    if save_csv:
        out = "example_data.csv"
        tbl.head(rows_to_show).to_csv(out, index=False)
        print(f"[INFO] Saved example table to {out}")

# -----------------------------
# Small utils
# -----------------------------
def sigmoid(z):
    z = np.clip(z, -40.0, 40.0)
    return 1.0 / (1.0 + np.exp(-z))

def bernoulli_entropy(p):
    p = np.clip(p, 1e-12, 1-1e-12)
    return -(p*np.log(p) + (1-p)*np.log(1-p))

def exp_grad_simplex_step(omega, grad, eta):
    new = omega * np.exp(-eta*grad)
    s = new.sum()
    return (new / s) if s > 0 else np.ones_like(omega)/len(omega)

def gaussian_pdf(x, mean, cov):
    d = x.shape[-1]
    # ensure PD
    try:
        L = np.linalg.cholesky(cov)
    except np.linalg.LinAlgError:
        cov = cov + 1e-6*np.eye(d)
        L = np.linalg.cholesky(cov)
    diff = x - mean
    sol = np.linalg.solve(L, diff)
    quad = np.dot(sol, sol)
    logdet = 2.0*np.log(np.diag(L)).sum()
    val = -0.5*(quad + logdet + d*math.log(2*math.pi))
    return float(np.exp(val))

# -----------------------------
# Data: real or synthetic
# -----------------------------
def load_or_make_data(cfg: Config) -> Tuple[pd.DataFrame, np.ndarray]:
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    if os.path.exists(cfg.CSV_PATH):
        df = pd.read_csv(cfg.CSV_PATH)
        required = {"cell_line", "drug", "dose", "viability"}
        if not required.issubset(df.columns):
            raise ValueError(f"CSV must contain columns {required}")
        if cfg.DRUG_NAME in df["drug"].unique():
            grid = np.sort(df.loc[df["drug"] == cfg.DRUG_NAME, "dose"].unique())
            return df, grid
        else:
            if cfg.VERBOSE:
                print(f"[WARN] Drug '{cfg.DRUG_NAME}' not found in CSV. Falling back to synthetic.")
    else:
        if cfg.VERBOSE:
            print(f"[WARN] CSV '{cfg.CSV_PATH}' not found. Generating synthetic dataset.")

    # Synthetic fallback
    grid = (np.array([1e-3, 3e-3, 1e-2, 3e-2, 1e-1, 3e-1, 1.0], dtype=float)
            if cfg.SYNTH_DOSE_GRID is None else np.array(cfg.SYNTH_DOSE_GRID, dtype=float))
    if cfg.SYNTH_THETA_MIX is None:
        mix_pi = np.array([0.5, 0.5])
        mix_mu = np.array([[-1.0, 6.0],
                           [ 0.5, 3.0]])
        mix_cov = np.array([[[0.25, 0.0],[0.0, 0.5]],
                            [[0.15, 0.0],[0.0, 0.4]]])
    else:
        mix_pi = cfg.SYNTH_THETA_MIX["pi"]
        mix_mu = cfg.SYNTH_THETA_MIX["mu"]
        mix_cov = cfg.SYNTH_THETA_MIX["cov"]

    rows, drug = [], cfg.DRUG_NAME
    cells = [f"CL_{i:04d}" for i in range(1, cfg.SYNTH_NUM_CELLS+1)]
    for cl in cells:
        j = rng.choice(len(mix_pi), p=mix_pi)
        theta = rng.multivariate_normal(mix_mu[j], mix_cov[j])
        a, b = theta
        for d in grid:
            p = sigmoid(a + b*np.log(d + cfg.EPS_DOSE))
            # viability proxy in [0,1] (mean of 3 Bernoullis -> smoother)
            y = rng.binomial(1, p, size=3).mean()
            viability = 1.0 - y
            rows.append((cl, drug, d, float(viability)))
    df = pd.DataFrame(rows, columns=["cell_line", "drug", "dose", "viability"])
    return df, grid

# -----------------------------
# Centering transform
# -----------------------------
def dose_offset_for_drug(df, drug, eps) -> float:
    g = np.sort(df.loc[df["drug"] == drug, "dose"].unique())
    return float(np.log(g + eps).mean())

def x_from_dose(doses, offset, eps):
    return np.log(doses + eps) - offset

# -----------------------------
# Logistic MLE/MAP (centered x)
# -----------------------------
def logistic_mle_map(x, y, l2_prec=1e-2, max_iter=100, tol=1e-8):
    X = np.column_stack([np.ones_like(x), x])  # [1, centered log-dose]
    theta = np.zeros(2, dtype=float)
    I = np.eye(2)
    for _ in range(max_iter):
        z = X @ theta
        p = sigmoid(z)
        W = p*(1-p)
        grad = X.T @ (p - y) + l2_prec*theta
        H = (X.T * W) @ X + l2_prec*I
        step = np.linalg.solve(H, grad)
        theta_new = theta - step
        if np.linalg.norm(step) < tol*(1+np.linalg.norm(theta)):
            theta = theta_new; break
        theta = theta_new
    # final Hessian
    z = X @ theta
    p = sigmoid(z)
    W = p*(1-p)
    H = (X.T * W) @ X + l2_prec*I
    return theta, H

# -----------------------------
# Empirical prior & LMA
# -----------------------------
def build_empirical_prior_thetas(df, drug, target_cell, bin_thresh, eps, x_offset):
    hist = df[(df["drug"] == drug) & (df["cell_line"] != target_cell)].copy()
    thetas = []
    for cl, g in hist.groupby("cell_line"):
        x = x_from_dose(g["dose"].values, x_offset, eps)
        y = (g["viability"].values <= bin_thresh).astype(float)
        theta_hat, _ = logistic_mle_map(x, y, l2_prec=cfg.L2_PREC,
                                        max_iter=cfg.MAX_ITERS, tol=cfg.TOL)
        thetas.append(theta_hat)
    thetas = np.array(thetas)
    return thetas, hist

def kmeans_simple(X, J, seed=0):
    rng = np.random.default_rng(seed)
    n, d = X.shape
    centers = X[rng.choice(n, size=J, replace=False)]
    for _ in range(50):
        dists = ((X[:, None, :] - centers[None, :, :])**2).sum(axis=2)
        labels = np.argmin(dists, axis=1)
        new_centers = np.vstack([X[labels==j].mean(axis=0) if np.any(labels==j) else centers[j] for j in range(J)])
        if np.allclose(new_centers, centers): break
        centers = new_centers
    return labels

def lma_from_empirical_samples(theta_samples, J, ridge, seed):
    labels = kmeans_simple(theta_samples, J, seed)
    mus, Sigmas, omegas = [], [], []
    for j in range(J):
        pts = theta_samples[labels==j]
        if len(pts) == 0:
            pts = theta_samples[np.random.choice(len(theta_samples), size=3, replace=True)]
        mu = pts.mean(axis=0)
        centered = pts - mu
        cov = (centered.T @ centered) / max(len(pts)-1, 1)
        # robust covariance floor
        cov = cov + ridge*np.eye(2)
        mus.append(mu); Sigmas.append(cov); omegas.append(len(pts))
    omegas = np.array(omegas, dtype=float); omegas /= omegas.sum()
    return np.array(mus), np.array(Sigmas), omegas

def refine_weights_reverse_kl(theta_support, mus, Sigmas, omega_init, iters, step):
    M, J = len(theta_support), len(omega_init)
    omega = omega_init.copy()
    # Precompute Gaussian φ_j(θ_m)
    Phi = np.zeros((M, J))
    for m in range(M):
        for j in range(J):
            Phi[m, j] = gaussian_pdf(theta_support[m], mus[j], Sigmas[j]) + 1e-300
    w_m = np.ones(M)/M
    for _ in range(iters):
        denom = Phi @ omega + 1e-300
        grad = - (w_m[:, None]*(Phi/denom[:, None])).sum(axis=0)
        omega = exp_grad_simplex_step(omega, grad, step)
    return omega

def mixture_moment_match(omegas, mus, Sigmas):
    mu_bar = (omegas[:, None] * mus).sum(axis=0)
    cov_bar = np.zeros((2,2))
    for j in range(len(omegas)):
        diff = (mus[j]-mu_bar).reshape(2,1)
        cov_bar += omegas[j]*(Sigmas[j] + diff@diff.T)
    return mu_bar, cov_bar

# -----------------------------
# EIG & design
# -----------------------------
def eig_terms_from_samples(theta_samples, dose_grid, x_offset, eps):
    S = len(theta_samples); K = len(dose_grid)
    alpha = theta_samples[:,0][:,None]; beta = theta_samples[:,1][:,None]
    xk = x_from_dose(dose_grid, x_offset, eps)[None, :]  # (1,K)
    P = sigmoid(alpha + beta * xk)                        # (S,K)
    pbar = P.mean(axis=0); Eh = bernoulli_entropy(P).mean(axis=0)
    return pbar, Eh

def greedy_design_gamma(pbar, Eh, N, min_per_dose=0, tau=0.0):
    """
    Greedy allocator with optional:
      - min_per_dose floor,
      - entropy regularization tau >= 0 (softly spreads gamma).
    """
    K = len(pbar)
    Delta = bernoulli_entropy(pbar) - Eh
    n = np.zeros(K, dtype=int)

    # enforce floor if requested
    used = 0
    if min_per_dose > 0:
        give = min(min_per_dose, max(N // K, 1))
        n += give
        used = give * K
        if used > N:  # safety
            n[:] = 0
            used = 0

    for _ in range(N - used):
        if tau > 0:
            denom = float(N)  # denominator for gamma
            gamma = n / denom
            eps = 1e-12
            delta_H = []
            for k in range(K):
                gk, gk_new = gamma[k], (n[k] + 1) / denom
                dH = -(gk_new * np.log(max(gk_new, eps)) - gk * np.log(max(gk, eps)))
                delta_H.append(dH)
            delta_H = np.array(delta_H)
            score = Delta + tau * N * delta_H
            kstar = int(np.argmax(score))
        else:
            kstar = int(np.argmax(Delta))
        n[kstar] += 1

    gamma = n / float(N)
    return n, gamma, Delta

def uniform_design(N: int, K: int) -> Tuple[np.ndarray, np.ndarray]:
    n = np.full(K, N // K, dtype=int)
    n[: (N % K)] += 1
    gamma = n / float(N)
    return n, gamma

def fisher_local_dopt_design(dose_grid, N, theta0, x_offset, eps) -> Tuple[np.ndarray, np.ndarray]:
    """
    Greedy local D-opt allocation at a fixed θ0 (alpha0, beta0) on the candidate grid.
    Each replicate adds A_k = p_k(1-p_k) * [[1, x_k],[x_k, x_k^2]] to the Fisher info.
    Choose k that maximizes det(M + A_k) at each step.
    """
    K = len(dose_grid)
    xk = x_from_dose(dose_grid, x_offset, eps)
    alpha0, beta0 = theta0
    p = sigmoid(alpha0 + beta0 * xk)
    w = p*(1-p)  # scalar weights per dose
    blocks = [w[k] * np.array([[1.0, xk[k]],[xk[k], xk[k]**2]]) for k in range(K)]

    M = 1e-8 * np.eye(2)  # small ridge to start
    n = np.zeros(K, dtype=int)
    for _ in range(N):
        best_k, best_logdet = 0, -np.inf
        for k in range(K):
            M_try = M + blocks[k]
            sign, logdet = np.linalg.slogdet(M_try)
            if sign <= 0: continue
            if logdet > best_logdet:
                best_logdet = logdet; best_k = k
        n[best_k] += 1
        M = M + blocks[best_k]
    gamma = n / float(N)
    return n, gamma

# -----------------------------
# Posterior Laplace (centered x) and EC50
# -----------------------------
def posterior_laplace_gaussian_prior(x, y, mu_prior, Sigma_prior, damp=1e-8):
    Prec = np.linalg.inv(Sigma_prior + damp*np.eye(2))
    X = np.column_stack([np.ones_like(x), x])
    theta = mu_prior.copy()
    for _ in range(100):
        z = X @ theta
        p = sigmoid(z)
        W = p*(1-p)
        grad_ll = X.T @ (y - p)
        grad = -( -grad_ll + Prec @ (mu_prior - theta) )
        H = (X.T * W) @ X + Prec
        step = np.linalg.solve(H, grad)
        theta_new = theta - step
        if np.linalg.norm(step) < 1e-8*(1+np.linalg.norm(theta)):
            theta = theta_new; break
        theta = theta_new
    # covariance
    z = X @ theta; p = sigmoid(z); W = p*(1-p)
    H = (X.T * W) @ X + Prec
    Sigma_post = np.linalg.inv(H + 1e-12*np.eye(2))
    return theta, Sigma_post

def ec50_from_theta(theta, x_offset):
    a, b = theta
    # model: α + β * (log d - x_offset); solve α + β*(log d - x_offset)=0
    return float(np.exp(x_offset - a / (b + 1e-12)))

def ec50_ci_delta(theta, Sigma, x_offset, z=1.96):
    a, b = theta
    g = lambda a,b: np.exp(x_offset - a/(b+1e-12))
    val = g(a,b)
    dga = -(1/(b+1e-12))*val
    dgb = (a/((b+1e-12)**2))*val
    grad = np.array([dga, dgb])
    var = float(grad @ Sigma @ grad)
    se = math.sqrt(max(var, 0.0))
    return val - z*se, val + z*se

def gaussian_entropy(Sigma):
    sign, logdet = np.linalg.slogdet(Sigma)
    if sign <= 0:
        return float("nan")
    d = Sigma.shape[0]
    return 0.5 * (d * (1 + math.log(2 * math.pi)) + logdet)

# -----------------------------
# Simulation helpers (CRNs)
# -----------------------------
def build_crn(dose_grid, n_max_per_dose, seed=123):
    rng = np.random.default_rng(seed)
    # map dose -> a long vector of uniforms we will reuse, one prefix per design
    return {float(d): rng.random(int(n_max_per_dose)) for d in dose_grid}

def simulate_data_crn(theta, dose_grid, n_counts, x_offset, eps, crn):
    rows = []
    for k, d in enumerate(dose_grid):
        nk = int(n_counts[k])
        if nk == 0:
            continue
        xk = x_from_dose(np.array([d]), x_offset, eps)[0]
        p = sigmoid(theta[0] + theta[1]*xk)
        u = crn[float(d)][:nk]  # reuse same uniforms for this dose
        ys = (u < p).astype(float)
        for y in ys:
            rows.append((d, y))
    return pd.DataFrame(rows, columns=["dose", "y"])

# -----------------------------
# Plotting helpers (matplotlib)
# -----------------------------
def plot_delta(dose_grid, Delta):
    plt.figure(figsize=(6,3))
    plt.bar(np.arange(len(dose_grid)), Delta)
    plt.xticks(np.arange(len(dose_grid)), [f"{d:g}" for d in dose_grid], rotation=0)
    plt.xlabel("Dose")
    plt.ylabel("Δ_k = h( p̄_k ) - E[h(P_{s,k})]")
    plt.title("Per-dose MI gain Δ_k")
    plt.tight_layout()

def plot_gamma_all(dose_grid, gammas_dict):
    plt.figure(figsize=(6,3))
    K = len(dose_grid)
    x = np.arange(K)
    width = 0.25
    keys = list(gammas_dict.keys())
    for i, name in enumerate(keys):
        plt.bar(x + (i - (len(keys)-1)/2)*width, gammas_dict[name], width, label=name)
    plt.xticks(x, [f"{d:g}" for d in dose_grid])
    plt.xlabel("Dose")
    plt.ylabel("Allocation γ_k")
    plt.title("Design allocations")
    plt.legend()
    plt.tight_layout()

def ellipse_points(mu, Sigma, nsig=1.0, num=200):
    # return (x,y) points of nsig Gaussian ellipse
    vals, vecs = np.linalg.eigh(Sigma)
    vals = np.maximum(vals, 1e-12)
    order = np.argsort(vals)[::-1]
    vals, vecs = vals[order], vecs[:, order]
    t = np.linspace(0, 2*np.pi, num)
    circle = np.stack([np.cos(t), np.sin(t)], axis=0)  # (2,num)
    scale = nsig * np.sqrt(vals)
    ell = (vecs @ (scale[:, None] * circle))
    return mu[0] + ell[0], mu[1] + ell[1]

def plot_post_contours(posteriors: Dict[str, Tuple[np.ndarray, np.ndarray]]):
    plt.figure(figsize=(5,5))
    for name, (mu, Sigma) in posteriors.items():
        x, y = ellipse_points(mu, Sigma, nsig=1.0)
        plt.plot(x, y, label=f"{name} (1σ)")
        x2, y2 = ellipse_points(mu, Sigma, nsig=2.0)
        plt.plot(x2, y2, linestyle="--", alpha=0.6)
    plt.xlabel(r"$\alpha$ (centered model)")
    plt.ylabel(r"$\beta$")
    plt.title("Posterior Gaussian contours")
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()

# ----- EDA plots -----
def plot_some_cell_curves(df, n_cells=8):
    """Example dose–viability curves for a subset of cell lines."""
    rng_local = np.random.default_rng(0)
    cells = df["cell_line"].unique()
    if len(cells) > n_cells:
        cells = list(rng_local.choice(cells, size=n_cells, replace=False))
    plt.figure(figsize=(6,4))
    for cl in cells:
        g = df[df["cell_line"]==cl].sort_values("dose")
        plt.plot(g["dose"], g["viability"], marker="o")
    plt.xscale("log")
    plt.xlabel("Dose (log scale)")
    plt.ylabel("Viability (0–1)")
    plt.title("Example dose–viability curves (synthetic)")
    plt.tight_layout()

def plot_response_rate_by_dose(df_with_y):
    """Binarized response rate (P(y=1)) as a function of dose."""
    agg = df_with_y.groupby("dose", as_index=False)["y"].mean()
    plt.figure(figsize=(6,4))
    plt.plot(agg["dose"], agg["y"], marker="o")
    plt.xscale("log")
    plt.ylim(0, 1)
    plt.xlabel("Dose (log scale)")
    plt.ylabel("Response rate  P(y=1)")
    plt.title("Binarized response rate vs dose")
    plt.tight_layout()

def plot_viability_summary(df):
    """Mean ± SE of viability across cell lines at each dose."""
    agg = df.groupby("dose")["viability"]
    means = agg.mean().values
    stds = agg.std(ddof=1).values
    ns = agg.count().values
    ses = stds/np.sqrt(np.maximum(ns, 1))
    doses = np.array(sorted(df["dose"].unique()))
    plt.figure(figsize=(6,4))
    plt.errorbar(doses, means, yerr=ses, fmt="-o")
    plt.xscale("log")
    plt.ylim(0, 1)
    plt.xlabel("Dose (log scale)")
    plt.ylabel("Mean viability  ±  SE")
    plt.title("Viability summary across cell lines")
    plt.tight_layout()

# -----------------------------
# Main
# -----------------------------
def main(cfg: Config):
    np.random.seed(cfg.RANDOM_SEED)
    df, dose_grid = load_or_make_data(cfg)

    # Print a small example table of the data we use
    print_example_data_table(df, cfg, n_cells=3, save_csv=True)

    # Ensure a valid target
    if cfg.TARGET_CELL not in df["cell_line"].unique():
        cfg.TARGET_CELL = df["cell_line"].iloc[0]
        if cfg.VERBOSE:
            print(f"[WARN] TARGET_CELL not found. Using {cfg.TARGET_CELL}")

    # Shared offset for the drug (stabilizes α,β)
    x_offset = dose_offset_for_drug(df, cfg.DRUG_NAME, cfg.EPS_DOSE)

    # --- EDA: quick visual summaries (save PNGs) ---
    df_eda = df.copy()
    df_eda["y"] = (df_eda["viability"] <= cfg.BIN_THRESH).astype(float)
    plot_some_cell_curves(df, n_cells=8)
    plt.savefig("eda_curves.png", dpi=150)
    plot_viability_summary(df)
    plt.savefig("eda_viability_summary.png", dpi=150)
    plot_response_rate_by_dose(df_eda)
    plt.savefig("eda_response_rate.png", dpi=150)

    # Historical θ-hats
    thetas_hist, _ = build_empirical_prior_thetas(df, cfg.DRUG_NAME, cfg.TARGET_CELL,
                                                  cfg.BIN_THRESH, cfg.EPS_DOSE, x_offset)
    if cfg.VERBOSE:
        print(f"[INFO] Historical θ-hats: {thetas_hist.shape[0]} cell lines")

    # LMA: cluster + covariance floor + reverse-KL weights
    mus, Sigmas, omega0 = lma_from_empirical_samples(thetas_hist, cfg.J, cfg.RIDGE_COV, cfg.KM_SEED)
    omega = refine_weights_reverse_kl(thetas_hist, mus, Sigmas, omega0,
                                      cfg.REVERSE_KL_ITERS, cfg.REVERSE_KL_STEP)
    if cfg.VERBOSE:
        print(f"[INFO] LMA components J={cfg.J}")
        for j in range(cfg.J):
            print(f"  j={j}: ω={omega[j]:.3f}, μ={mus[j].round(3)}, diagΣ={np.diag(Sigmas[j]).round(3)}")

    # LMA samples for EIG terms
    rng = np.random.default_rng(cfg.RANDOM_SEED)
    comp = rng.choice(cfg.J, size=cfg.S, p=omega)
    theta_samps = np.vstack([rng.multivariate_normal(mus[j], Sigmas[j]) for j in comp])

    # EIG terms and DIAGNOSTICS
    pbar_k, Eh_k = eig_terms_from_samples(theta_samps, dose_grid, x_offset, cfg.EPS_DOSE)
    Delta = bernoulli_entropy(pbar_k) - Eh_k
    print("[DEBUG] pbar_k:", np.round(pbar_k, 4))
    print("[DEBUG] Eh_k  :", np.round(Eh_k, 4))
    print("[DEBUG] Delta :", np.round(Delta, 4))

    # Designs:
    # (A) LMA-EIG greedy (with min-per-dose floor and entropy regularization tau)
    n_LMA, gamma_LMA, Delta_check = greedy_design_gamma(
        pbar_k, Eh_k, cfg.BUDGET_N, cfg.MIN_PER_DOSE, cfg.ENTROPY_TAU
    )
    assert np.allclose(Delta, Delta_check)
    if cfg.VERBOSE:
        print(f"[INFO] Greedy γ (LMA-EIG): {gamma_LMA.round(3)}  (N={cfg.BUDGET_N}, doses={dose_grid})")

    # (B) Uniform baseline
    n_UNI, gamma_UNI = uniform_design(cfg.BUDGET_N, len(dose_grid))

    # (C) Fisher local D-opt baseline at θ0 = mixture mean (moment-matched)
    mu_prior, Sigma_prior = mixture_moment_match(omega, mus, Sigmas)
    n_DOPT, gamma_DOPT = fisher_local_dopt_design(dose_grid, cfg.BUDGET_N, mu_prior,
                                                  x_offset, cfg.EPS_DOSE)

    # EIG estimates (linear in γ)
    EIG_LMA = float((n_LMA * Delta).sum())
    EIG_UNI = float((n_UNI * Delta).sum())
    EIG_DOPT = float((n_DOPT * Delta).sum())

    # Target line data and its θ̂ for simulation ground-truth
    gtar = df[(df["drug"] == cfg.DRUG_NAME) & (df["cell_line"] == cfg.TARGET_CELL)].copy()
    if len(gtar) == 0: raise ValueError("No target cell-line data found.")
    x_tar = x_from_dose(gtar["dose"].values, x_offset, cfg.EPS_DOSE)
    y_tar_bin = (gtar["viability"].values <= cfg.BIN_THRESH).astype(float)
    theta_tar_hat, _ = logistic_mle_map(x_tar, y_tar_bin, l2_prec=cfg.L2_PREC,
                                        max_iter=cfg.MAX_ITERS, tol=cfg.TOL)

    # CRNs for fair simulation across designs
    n_max = int(max(n_LMA.max(), n_UNI.max(), n_DOPT.max()))
    crn = build_crn(dose_grid, n_max, seed=123)

    # Simulate prospective outcomes for each design (CRNs)
    sim_LMA  = simulate_data_crn(theta_tar_hat, dose_grid, n_LMA,  x_offset, cfg.EPS_DOSE, crn)
    sim_UNI  = simulate_data_crn(theta_tar_hat, dose_grid, n_UNI,  x_offset, cfg.EPS_DOSE, crn)
    sim_DOPT = simulate_data_crn(theta_tar_hat, dose_grid, n_DOPT, x_offset, cfg.EPS_DOSE, crn)

    # Posterior Laplace for each design (Gaussian prior from LMA moment match)
    def fit_post(sim_df):
        x = x_from_dose(sim_df["dose"].values, x_offset, cfg.EPS_DOSE)
        y = sim_df["y"].values
        return posterior_laplace_gaussian_prior(x, y, mu_prior, Sigma_prior, cfg.L2_DAMP)

    theta_post_LMA, Sigma_post_LMA = fit_post(sim_LMA)
    theta_post_UNI, Sigma_post_UNI = fit_post(sim_UNI)
    theta_post_DOPT, Sigma_post_DOPT = fit_post(sim_DOPT)

    # Metrics
    def summarize(name, n_counts, gamma, theta_post, Sigma_post, EIG_hat):
        H_post = gaussian_entropy(Sigma_post)
        ec50 = ec50_from_theta(theta_post, x_offset)
        lo, hi = ec50_ci_delta(theta_post, Sigma_post, x_offset)
        print(f"\n=== {name} ===")
        print(f"counts n_k: {n_counts} (gamma={gamma.round(3)})")
        print(f"EIG_est: {EIG_hat:.6f}")
        print(f"Posterior MAP θ: {theta_post.round(4)}  |  diag(Σ_post): {np.diag(Sigma_post).round(4)}")
        print(f"Posterior entropy H: {H_post:.4f}")
        print(f"EC50: {ec50:.4g}  CI95%: [{lo:.4g}, {hi:.4g}]")
        return H_post, ec50, (lo, hi)

    print("\n=== RESULTS ===")
    print(f"Dose grid: {dose_grid}")

    H_LMA, EC50_LMA, CI_LMA = summarize("LMA–EIG", n_LMA, gamma_LMA, theta_post_LMA, Sigma_post_LMA, EIG_LMA)
    H_UNI,  EC50_UNI,  CI_UNI  = summarize("Uniform", n_UNI, gamma_UNI, theta_post_UNI, Sigma_post_UNI, EIG_UNI)
    H_DOPT, EC50_DOPT, CI_DOPT = summarize("Fisher D-opt (local)", n_DOPT, gamma_DOPT, theta_post_DOPT, Sigma_post_DOPT, EIG_DOPT)

    # Plug-in log-lik on existing target observations (for LMA design only; replicate for others if desired)
    p_tar = sigmoid(theta_post_LMA[0] + theta_post_LMA[1]*x_tar)
    p_tar = np.clip(p_tar, 1e-8, 1-1e-8)
    ll = (y_tar_bin*np.log(p_tar) + (1-y_tar_bin)*np.log(1-p_tar)).sum()
    print(f"\nPlug-in log-likelihood on target line (LMA posterior): {ll:.3f}")

    # -----------------------------
    # Plots + save figures
    # -----------------------------
    plot_delta(dose_grid, Delta)
    plt.savefig("delta_per_dose.png", dpi=150)

    gammas_all = {
        "LMA–EIG": gamma_LMA,
        "Uniform": gamma_UNI,
        "D-opt": gamma_DOPT,
    }
    plot_gamma_all(dose_grid, gammas_all)
    plt.savefig("design_allocations.png", dpi=150)

    posteriors = {
        "LMA–EIG": (theta_post_LMA, Sigma_post_LMA),
        "Uniform": (theta_post_UNI, Sigma_post_UNI),
        "D-opt": (theta_post_DOPT, Sigma_post_DOPT),
    }
    plot_post_contours(posteriors)
    plt.savefig("posterior_contours.png", dpi=150)

    plt.show()

if __name__ == "__main__":
    main(cfg)

"""# end."""
