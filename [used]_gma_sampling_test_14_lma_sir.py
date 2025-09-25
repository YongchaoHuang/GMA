# -*- coding: utf-8 -*-
"""[used] GMA sampling test 14: LMA-SIR.ipynb

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

#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ========= args =========
import sys
sys.argv = [
    "notebook",
    "--start", "2020-03-01",
    "--end",   "2020-04-15",
    "--smooth7",
    "--nstarts", "6",
    "--samples", "1000",
    "--max_days", "21",
    "--stride", "2",
]
# ===========================================================

"""
SIR + Laplace-mixture inference for England COVID wave-1 (March–May 2020)
---------------------------------------------------------------------------
- Fetches daily NEW CASES by specimen date for England from the UKHSA dashboard API.
- Fits a simple SIR model with Poisson observation on daily incidence (−ΔS_t)
  and a reporting fraction ρ.
- Uses multi-start optimisation to find posterior modes of θ=(β, γ, I0, ρ),
  builds a Laplace mixture (means, covariances, weights) via local evidence,
  and draws posterior samples for R0 = β/γ by stratified sampling.

Requirements: numpy, pandas, scipy, requests, matplotlib
"""

from __future__ import annotations
import argparse, math, sys, json, time
from dataclasses import dataclass
from typing import Callable, Tuple, List, Dict

import numpy as np
import pandas as pd
import requests
from scipy import optimize, integrate
import matplotlib.pyplot as plt

# ---------------------------
# 1) Data loading (UKHSA API)
# ---------------------------
UKHSA_CASES_ENDPOINT = (
    "https://api.ukhsa-dashboard.data.gov.uk/"
    "themes/infectious_disease/sub_themes/respiratory/topics/COVID-19/"
    "geography_types/Nation/geographies/England/metrics/COVID-19_cases_casesByDay"
)

def fetch_england_cases(year: int = 2020, page_size: int = 365) -> pd.DataFrame:
    """Fetch daily new cases (by specimen date) for England for a given year.
    Returns DataFrame with columns ['date', 'cases'].
    """
    params = {"format": "json", "page_size": page_size, "year": year}
    r = requests.get(UKHSA_CASES_ENDPOINT, params=params, timeout=30)
    r.raise_for_status()
    payload = r.json()
    results = payload.get("results", [])
    df = pd.DataFrame(results)
    if df.empty:
        raise RuntimeError("Empty response from UKHSA API. Try a different year or check connectivity.")
    df = df[["date", "metric_value"]].rename(columns={"metric_value": "cases"})
    df["date"] = pd.to_datetime(df["date"])  # already daily
    df = df.sort_values("date").reset_index(drop=True)
    return df

# ---------------------------------
# 2) Simple SIR simulator + mapping
# ---------------------------------
@dataclass
class SIRParams:
    beta: float   # transmission rate per day
    gamma: float  # removal rate per day
    I0: float     # initial infectious count
    rho: float    # reporting fraction in (0,1]

def sir_ode(t: float, y: np.ndarray, beta: float, gamma: float, N: float) -> List[float]:
    S, I, R = y
    dS = -beta * S * I / N
    dI = -dS - gamma * I
    dR = gamma * I
    return [dS, dI, dR]

def simulate_sir_daily(theta: SIRParams, N: float, dates: np.ndarray, dense_dt: float = 1.0) -> Dict[str, np.ndarray]:
    """Solve SIR and return states at daily grid and model-implied daily incidence.
    Incidence (new infections) is approximated by −ΔS over each day.
    """
    beta, gamma, I0, rho = theta.beta, theta.gamma, theta.I0, theta.rho
    t_eval = np.arange(0, len(dates), 1.0)

    S0 = max(N - I0, 1.0)
    I0 = max(I0, 1e-6)
    y0 = [S0, I0, 0.0]

    def rhs(t, y):
        return sir_ode(t, y, beta, gamma, N)

    sol = integrate.solve_ivp(
        rhs,
        t_span=(0.0, float(t_eval[-1])),
        y0=y0,
        t_eval=t_eval,
        method="RK45",
        max_step=dense_dt,
        rtol=1e-6,
        atol=1e-8,
    )
    if not sol.success:
        raise RuntimeError(f"ODE solver failed: {sol.message}")

    S, I, R = sol.y
    dS = np.diff(S, prepend=S[0])
    incidence = np.maximum(-dS, 1e-12)
    return {"S": S, "I": I, "R": R, "incidence": incidence}

# ----------------------------
# 3) Priors & log-posterior
# ----------------------------
@dataclass
class Priors:
    ln_beta_mu: float = math.log(0.35)
    ln_beta_sigma: float = 0.5
    ln_gamma_mu: float = math.log(1/7.0)
    ln_gamma_sigma: float = 0.5
    ln_I0_mu: float = math.log(100.0)
    ln_I0_sigma: float = 1.0
    rho_a: float = 2.0
    rho_b: float = 8.0

def log_prior(theta: SIRParams, pri: Priors) -> float:
    beta, gamma, I0, rho = theta.beta, theta.gamma, theta.I0, theta.rho
    if beta <= 0 or gamma <= 0 or I0 <= 0 or not (0 < rho <= 1):
        return -np.inf
    lp = 0.0
    def llognorm(x, mu, sig):
        return -((math.log(x) - mu) ** 2) / (2 * sig ** 2) - math.log(x) - math.log(sig * math.sqrt(2 * math.pi))
    lp += llognorm(beta, pri.ln_beta_mu, pri.ln_beta_sigma)
    lp += llognorm(gamma, pri.ln_gamma_mu, pri.ln_gamma_sigma)
    lp += llognorm(I0, pri.ln_I0_mu, pri.ln_I0_sigma)
    lp += (pri.rho_a - 1) * math.log(rho) + (pri.rho_b - 1) * math.log(1 - rho) - (
        math.lgamma(pri.rho_a) + math.lgamma(pri.rho_b) - math.lgamma(pri.rho_a + pri.rho_b)
    )
    return lp

def log_likelihood(theta: SIRParams, N: float, dates: np.ndarray, cases: np.ndarray) -> float:
    sim = simulate_sir_daily(theta, N=N, dates=dates)
    lam = theta.rho * sim["incidence"]
    lam = np.maximum(lam, 1e-9)
    y = cases.astype(float)
    from scipy.special import gammaln
    ll = np.sum(y * np.log(lam) - lam - gammaln(y + 1.0))
    return float(ll)

def log_posterior(theta: SIRParams, N: float, dates: np.ndarray, cases: np.ndarray, pri: Priors) -> float:
    lp = log_prior(theta, pri)
    if not np.isfinite(lp):
        return -np.inf
    ll = log_likelihood(theta, N, dates, cases)
    return lp + ll

# -----------------------------------------
# 4) Numerical gradients / Hessians (FD)
# -----------------------------------------
def _theta_to_array(theta: SIRParams) -> np.ndarray:
    return np.array([theta.beta, theta.gamma, theta.I0, theta.rho], dtype=float)

def _array_to_theta(x: np.ndarray) -> SIRParams:
    return SIRParams(beta=float(x[0]), gamma=float(x[1]), I0=float(x[2]), rho=float(x[3]))

def fd_hessian(f: Callable[[np.ndarray], float], x0: np.ndarray, h: np.ndarray) -> np.ndarray:
    n = x0.size
    H = np.zeros((n, n), dtype=float)
    f0 = f(x0)
    for i in range(n):
        ei = np.zeros(n); ei[i] = 1.0
        f_plus  = f(x0 + h[i] * ei)
        f_minus = f(x0 - h[i] * ei)
        H[i, i] = (f_plus - 2.0 * f0 + f_minus) / (h[i] ** 2)
    for i in range(n):
        for j in range(i + 1, n):
            ei = np.zeros(n); ei[i] = 1.0
            ej = np.zeros(n); ej[j] = 1.0
            f_pp = f(x0 + h[i] * ei + h[j] * ej)
            f_pm = f(x0 + h[i] * ei - h[j] * ej)
            f_mp = f(x0 - h[i] * ei + h[j] * ej)
            f_mm = f(x0 - h[i] * ei - h[j] * ej)
            val = (f_pp - f_pm - f_mp + f_mm) / (4.0 * h[i] * h[j])
            H[i, j] = H[j, i] = val
    return H

def nearest_pd(A: np.ndarray, jitter: float = 1e-8) -> np.ndarray:
    w, V = np.linalg.eigh((A + A.T) / 2.0)
    w_clipped = np.maximum(w, jitter)
    return (V * w_clipped) @ V.T

# ---------------------------------
# 5) Optimisation (multi-start)
# ---------------------------------
def sample_from_priors(rng: np.random.Generator, pri: Priors) -> SIRParams:
    def lognormal(mu, sigma):
        return float(np.exp(rng.normal(mu, sigma)))
    beta  = lognormal(pri.ln_beta_mu, pri.ln_beta_sigma)
    gamma = lognormal(pri.ln_gamma_mu, pri.ln_gamma_sigma)
    I0    = lognormal(pri.ln_I0_mu, pri.ln_I0_sigma)
    rho   = float(rng.beta(pri.rho_a, pri.rho_b))
    return SIRParams(beta, gamma, I0, rho)

def optimise_mode(theta0: SIRParams, N: float, dates: np.ndarray, cases: np.ndarray, pri: Priors) -> Tuple[SIRParams, float]:
    def to_x(th: SIRParams) -> np.ndarray:
        b, g, I0, r = th.beta, th.gamma, th.I0, th.rho
        r = np.clip(r, 1e-6, 1 - 1e-6)
        return np.array([math.log(b), math.log(g), math.log(I0), math.log(r) - math.log(1 - r)], dtype=float)
    def from_x(x: np.ndarray) -> SIRParams:
        b = math.exp(x[0]); g = math.exp(x[1]); I0 = math.exp(x[2]); r = 1.0 / (1.0 + math.exp(-x[3]))
        return SIRParams(b, g, I0, r)
    def nlp(x: np.ndarray) -> float:
        th = from_x(x); return -log_posterior(th, N, dates, cases, pri)
    x0 = to_x(theta0)
    res = optimize.minimize(nlp, x0, method="L-BFGS-B")
    th_hat = from_x(res.x)
    lp_hat = log_posterior(th_hat, N, dates, cases, pri)
    return th_hat, lp_hat

def dedupe_modes(modes: List[SIRParams], lps: List[float], tol: Dict[str, float] | None = None) -> Tuple[List[SIRParams], List[float]]:
    if tol is None: tol = {"beta": 1e-3, "gamma": 1e-3, "I0": 1.0, "rho": 1e-3}
    kept_modes: List[SIRParams] = []; kept_lps: List[float] = []
    for th, lp in sorted(zip(modes, lps), key=lambda z: -z[1]):
        dup = False
        for th2 in kept_modes:
            if (abs(th.beta - th2.beta) <= tol["beta"] * max(1.0, th2.beta)
                and abs(th.gamma - th2.gamma) <= tol["gamma"] * max(1.0, th2.gamma)
                and abs(th.I0 - th2.I0) <= tol["I0"]
                and abs(th.rho - th2.rho) <= tol["rho"]):
                dup = True; break
        if not dup:
            kept_modes.append(th); kept_lps.append(lp)
    return kept_modes, kept_lps

# --------------------------------------------
# 6) Laplace mixture + sampling of R0
# --------------------------------------------
@dataclass
class LaplaceComponent:
    mu: np.ndarray
    Sigma: np.ndarray
    weight: float
    lp_at_mode: float

def build_laplace_mixture(modes: List[SIRParams], lps: List[float], N: float, dates: np.ndarray, cases: np.ndarray, pri: Priors,
                           kappa: float = 1.0) -> List[LaplaceComponent]:
    comps: List[LaplaceComponent] = []
    d = 4
    for th, lp in zip(modes, lps):
        mu = _theta_to_array(th)
        def f_theta(xvec: np.ndarray) -> float:
            thx = _array_to_theta(xvec); return log_posterior(thx, N, dates, cases, pri)
        h = np.array([max(1e-4, 1e-3*mu[0]), max(1e-4, 1e-3*mu[1]), max(1e-2, 1e-3*mu[2]), 1e-3], dtype=float)
        H = fd_hessian(f_theta, mu, h)
        Prec = nearest_pd(-H, jitter=1e-6)
        Sigma = np.linalg.inv(Prec)
        if kappa != 1.0: Sigma = (kappa**2) * Sigma
        comps.append(LaplaceComponent(mu=mu, Sigma=Sigma, weight=np.nan, lp_at_mode=lp))
    wtilde = []
    for c in comps:
        sign, logdet = np.linalg.slogdet(c.Sigma)
        if sign <= 0: logdet = np.log(np.finfo(float).tiny)
        wtilde.append(c.lp_at_mode + 0.5*d*math.log(2*math.pi) + 0.5*logdet)
    wtilde = np.array(wtilde); m = np.max(wtilde)
    ws = np.exp(wtilde - m); ws = ws/np.sum(ws)
    for c, w in zip(comps, ws): c.weight = float(w)
    return comps

def sample_from_mixture(comps: List[LaplaceComponent], nsamples: int, rng: np.random.Generator) -> np.ndarray:
    weights = np.array([c.weight for c in comps], dtype=float)
    Js = rng.choice(len(comps), size=nsamples, p=weights)
    draws = np.zeros((nsamples, 4), dtype=float)
    for i, j in enumerate(Js):
        c = comps[j]; ok = False; tries = 0
        while not ok and tries < 100:
            x = rng.multivariate_normal(mean=c.mu, cov=c.Sigma)
            b, g, I0, r = x
            if b > 0 and g > 0 and I0 > 0 and 0 < r < 1: ok = True; draws[i] = x
            tries += 1
        if not ok: draws[i] = c.mu
    return draws

# ----------------------------
# main (not guarded; executes in notebook cell)
# ----------------------------
ap = argparse.ArgumentParser()
ap.add_argument("--start", type=str, default="2020-03-01")
ap.add_argument("--end", type=str, default="2020-05-31")
ap.add_argument("--N", type=float, default=56550138)
ap.add_argument("--nstarts", type=int, default=16)
ap.add_argument("--seed", type=int, default=111)
ap.add_argument("--smooth7", action="store_true")
ap.add_argument("--kappa", type=float, default=1.5)
ap.add_argument("--samples", type=int, default=5000)
ap.add_argument("--max_days", type=int, default=21)
ap.add_argument("--stride", type=int, default=2)
args, _ = ap.parse_known_args()

rng = np.random.default_rng(args.seed)

# Data
df2020 = fetch_england_cases(2020, page_size=366)
df_list = [df2020]
if pd.to_datetime(args.end) >= pd.Timestamp("2021-01-01"):
    df2021 = fetch_england_cases(2021, page_size=366); df_list.append(df2021)
df = pd.concat(df_list).drop_duplicates("date").sort_values("date").reset_index(drop=True)

start, end = pd.to_datetime(args.start), pd.to_datetime(args.end)
msk = (df["date"] >= start) & (df["date"] <= end)
df = df.loc[msk, ["date", "cases"]].copy()
if df.empty: raise RuntimeError("No data in the specified window. Adjust --start/--end.")
if args.smooth7:
    df["cases"] = df["cases"].rolling(7, min_periods=1, center=True).mean()

if args.max_days is not None: df = df.head(int(args.max_days))
if args.stride and args.stride > 1: df = df.iloc[::int(args.stride)].reset_index(drop=True)

dates = df["date"].to_numpy()
cases = df["cases"].to_numpy()

print(f"Using T={len(dates)} daily observations after preprocessing.")

pri = Priors()

# =======================
# TIMERS: start inference
# =======================
t0 = time.perf_counter()

# Optimisation (multi-start)
modes: List[SIRParams] = []; lps: List[float] = []
for s in range(args.nstarts):
    th0 = sample_from_priors(rng, pri)
    try:
        th_hat, lp_hat = optimise_mode(th0, args.N, dates, cases, pri)
        if np.isfinite(lp_hat): modes.append(th_hat); lps.append(lp_hat)
    except Exception:
        continue

if not modes: raise RuntimeError("Optimisation failed for all starts; broaden priors or adjust window.")
modes, lps = dedupe_modes(modes, lps)
t1 = time.perf_counter()
print(f"Found {len(modes)} unique modes. Top-3 (β, γ, I0, ρ) and R0=β/γ:")
for i, (th, lp) in enumerate(list(zip(modes, lps))[:len(modes)]):
    print(f"  #{i+1}: beta={th.beta:.3f}, gamma={th.gamma:.3f}, I0={th.I0:.1f}, rho={th.rho:.3f}, R0={th.beta/th.gamma:.2f}, logpost={lp:.1f}")

# Laplace mixture
comps = build_laplace_mixture(modes, lps, args.N, dates, cases, pri, kappa=args.kappa)
W = np.array([c.weight for c in comps])
t2 = time.perf_counter()
print("Mixture weights:", np.round(W, 3))

# Posterior sampling
draws = sample_from_mixture(comps, nsamples=args.samples, rng=rng)
beta = draws[:, 0]; gamma = draws[:, 1]
R0 = beta / gamma
ci = np.percentile(R0, [2.5, 50, 97.5])
t3 = time.perf_counter()
print(f"R0 posterior: median {ci[1]:.2f} (95% CI {ci[0]:.2f}–{ci[2]:.2f})")

# Quick figure: data vs fitted mean incidence (top mode)
jmax = int(np.argmax(W))
th_star = _array_to_theta(comps[jmax].mu)
sim = simulate_sir_daily(th_star, args.N, dates)
lam = th_star.rho * sim["incidence"]

# fig = plt.figure(figsize=(9, 4))
# ax = fig.add_subplot(111)
# ax.plot(df["date"], df["cases"], "k*-", label="Observed cases", linewidth=2, markersize=8, zorder=3)
# ax.plot(df["date"], lam, color='green', label="Top-mode MAP trajectory (ρ·incidence)")
# ax.set_title("England COVID-19 daily new cases — SIR fit (LMA)")
# ax.set_xlabel("Date"); ax.set_ylabel("Cases per day")
# ax.legend(); fig.tight_layout()
# plt.show()

# unpack and derive
beta_s, gamma_s, I0_s, rho_s = draws.T
R0_s = beta_s / gamma_s

# assemble summary table
summary = pd.DataFrame({
    "mean":   [beta_s.mean(), gamma_s.mean(), R0_s.mean(), rho_s.mean(), I0_s.mean()],
    "median": [np.median(beta_s), np.median(gamma_s), np.median(R0_s),
               np.median(rho_s), np.median(I0_s)],
}, index=[r"beta (β)", r"gamma (γ)", r"R0 (β/γ)", r"rho (ρ)", r"I0"])

# print nicely
print(summary.round(3))

# (optional) plain prints
for name, arr in [(r"β", beta_s), (r"γ", gamma_s), (r"R₀", R0_s), (r"ρ", rho_s), (r"I₀", I0_s)]:
    print(f"{name:>4}: mean={arr.mean():.3f}, median={np.median(arr):.3f}")

# =========================
# 1) Posterior histograms (one row, order: β, γ, R0, ρ, I0)
# =========================
beta_s, gamma_s, I0_s, rho_s = draws.T
R0_s = beta_s / gamma_s

fig, axes = plt.subplots(1, 5, figsize=(18, 3.6))
def _hist(ax, data, title):
    ax.hist(data, bins=30, alpha=0.75, edgecolor="white")
    med = np.median(data)
    ax.axvline(med, linestyle="--", linewidth=2, label=f"median = {med:.3g}")
    ax.set_title(title, fontsize=11)
    ax.tick_params(axis="both", labelsize=9)

_hist(axes[0], beta_s, r"$\beta$ (per day)")
_hist(axes[1], gamma_s, r"$\gamma$ (per day)")
_hist(axes[2], R0_s,   r"$R_0=\beta/\gamma$")
_hist(axes[3], rho_s,  r"$\rho$ (reporting)")
_hist(axes[4], I0_s,   r"$I_0$ (initial infectious)")

# keep a single legend to avoid clutter
axes[2].legend(loc="upper right", fontsize=9)

fig.suptitle("Posterior samples (Laplace-mixture)", y=1.05)
fig.tight_layout()
plt.show()

# ==========================================
# 2) Predictive mean + uncertainty intervals
#    (simulate ALL posterior draws)
# ==========================================
T = len(dates)
lam_samples = np.zeros((draws.shape[0], T), dtype=float)
for k in range(draws.shape[0]):
    th = _array_to_theta(draws[k])
    sim_k = simulate_sir_daily(th, args.N, dates)
    lam_samples[k, :] = th.rho * sim_k["incidence"]

q_lo, q_med, q_hi = np.percentile(lam_samples, [2.5, 50, 97.5], axis=0)
lam_mean = lam_samples.mean(axis=0)
t4 = time.perf_counter()

fig = plt.figure(figsize=(10, 4))
ax = fig.add_subplot(111)

# Spaghetti of ALL posterior-draw mean trajectories
for k in range(lam_samples.shape[0]):
    if k == 0:
        ax.plot(df["date"], lam_samples[k], linewidth=0.8, alpha=0.08,
                label="all posterior draws of $\lambda_t$ (no noise)", zorder=0.5)
    else:
        ax.plot(df["date"], lam_samples[k], linewidth=0.8, alpha=0.08, zorder=0.5)

# Observations and summaries
reds = plt.cm.get_cmap("Reds")  # sequential reds
c_med  = reds(0.35)  # darkest
c_mean = reds(0.65)  # mid
c_top  = reds(0.95)  # lighter
# make the band match the family too
band_color = reds(0.30)

ax.plot(df["date"], df["cases"], "k*-", label="Observed cases", linewidth=3, markersize=8, zorder=3)
ax.fill_between(
    df["date"], q_lo, q_hi,
    color="0.95", alpha=0.35, edgecolor="grey",
    label="95% predictive band", zorder=0
)
ax.plot(df["date"], q_med,  linestyle="-.", linewidth=3, color=c_med, label="Posterior median", zorder=3)
ax.plot(df["date"], lam_mean, linestyle="-.", linewidth=3, color=c_mean, label="Posterior mean", zorder=3)
ax.plot(df["date"], lam,      linestyle="-.", linewidth=3, color=c_top, label="Top-mode mean", zorder=3)
ax.set_title("SIR predictions with posterior uncertainty")
ax.set_xlabel("Date"); ax.set_ylabel("Cases per day")
ax.legend()
# remove the first x-tick to avoid overlap
fig.canvas.draw()                 # ensure ticks are computed
ticks = ax.get_xticks()
ax.set_xticks(ticks[1:])          # keep all but the first
fig.tight_layout(); plt.show()

# =======================
# TIMERS: print summary
# =======================
opt_s  = t1 - t0
mix_s  = t2 - t1
samp_s = t3 - t2
pred_s = t4 - t3
tot_s  = t4 - t0
print(f"[Timing] optimisation={opt_s:.3f}s | Laplace-mixture={mix_s:.3f}s | sampling={samp_s:.3f}s | predictive={pred_s:.3f}s | total={tot_s:.3f}s")

"""# Compare our estimated reproduction number with existing literature."""

# === Static R0 vertical plot (ours in Blues; distinct colors for each reference; Rt=1 in legend) ===
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
import matplotlib.colors as mcolors

# Posterior samples for R0 (ours) — requires `draws` from your LMA cell
R0_s = draws[:, 0] / draws[:, 1]

# Literature references (label, central value, (low, high) or None, color)
lit_refs = [
    (r"Imperial R9 baseline",     2.4,  (2.0, 2.6),  "tab:red"),
    (r"Flaxman et al. (initial)", 3.8,  (2.4, 5.6),  "tab:green"),
    (r"Jarvis pre-intervention",  2.6,  None,        "tab:purple"),
    (r"Jarvis lockdown",          0.62, (0.37, 0.89), "tab:orange"),
]

fig = plt.figure(figsize=(5.4, 4.2))
ax = fig.add_subplot(111)

# Color family for ours (Blues)
blues = plt.cm.get_cmap("Blues")
ours_face = blues(0.85)
ours_edge = blues(0.55)
ours_rug  = blues(0.55)

# Boxplot (ours)
bp = ax.boxplot([R0_s], vert=True, widths=0.35, patch_artist=True,
                showfliers=False, medianprops=dict(color='k', lw=2))
box = bp['boxes'][0]
box.set(facecolor=ours_face, edgecolor=ours_edge, linewidth=2)
for w in bp['whiskers'] + bp['caps']:
    w.set(color=ours_edge, linewidth=2)

# Optional rug (subsample) to show sample spread
rng = np.random.default_rng(0)
idx = rng.choice(len(R0_s), size=min(300, len(R0_s)), replace=False)
ax.scatter(np.ones_like(idx) + 0.02 * rng.standard_normal(len(idx)),
           R0_s[idx], s=8, alpha=0.18, color=ours_rug, edgecolors='none', zorder=1)

# Reference lines/ribbons (distinct colors)
ref_line_handles, ref_labels = [], []
for (lab, mid, band, col) in lit_refs:
    h = ax.axhline(mid, color=col, lw=2.2, zorder=2)
    if band is not None:
        lo, hi = band
        ax.axhspan(lo, hi, color=mcolors.to_rgba(col, alpha=0.12), zorder=0)
        label_text = f"{lab}: {mid:g} [{lo:g}–{hi:g}]"
    else:
        label_text = f"{lab}: {mid:g}"
    ref_line_handles.append(Line2D([0], [0], color=col, lw=2.2))
    ref_labels.append(label_text)

# Rt=1 dashed threshold (IN legend)
rt1_handle = ax.axhline(1.0, color="k", lw=1.5, ls=":", label=r"$R_t=1$ threshold", zorder=1)

# Axes cosmetics
ax.set_xlim(0.5, 1.5)
ax.set_xticks([1]); ax.set_xticklabels([r"LMA $R_0$"])
ax.set_ylabel(r"$R_0$")
ax.set_title(r"Baseline reproduction number $R_0$ (posterior) vs. literature")
y_cap = float(np.nanpercentile(R0_s, 97))
ax.set_ylim(0, max(8.0, y_cap))
ax.grid(axis='y', alpha=0.3)

# Legend: ours + references + Rt=1 threshold
ours_proxy = Patch(facecolor=ours_face, edgecolor=ours_edge, label="LMA $R_0$ posterior")
ax.legend([ours_proxy] + ref_line_handles + [rt1_handle],
          ["LMA $R_0$ posterior"] + ref_labels + [r"$R_t=1$ threshold"],
          loc="upper left", fontsize=8, frameon=False)

fig.tight_layout(); plt.show()

"""# end."""
