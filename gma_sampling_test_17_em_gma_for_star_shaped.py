# -*- coding: utf-8 -*-
"""GMA sampling test 17: EM-GMA for star-shaped.ipynb

#yongchao.huang@abdn.ac.uk
"""

!lscpu

import psutil

print("CPU cores:", psutil.cpu_count(logical=False))
print("Logical CPUs:", psutil.cpu_count(logical=True))
print("Memory (GB):", round(psutil.virtual_memory().total / 1e9, 2))
print("Disk space (GB):", round(psutil.disk_usage('/').total / 1e9, 2))

"""# Begin.

# Compare 8 inference methods:

1. EM-GMA \\
2. MH \\
3. HMC (NUTS) \\
4. LMC \\
5. SVGD \\
6. MFVI-ADVI \\
7. GM-ADVI \\
8. EVI \\
"""

# compare_star_emgma_and_baselines_evi.py
# ------------------------------------------------------------
# Star-shaped target (mixture of rotated anisotropic Gaussians)
# + EM-GMA (population EM with SNIS; data-free)
# + Baselines: MH, NUTS(HMC, PyMC), LMC, SVGD, MFVI-ADVI, GM-ADVI (stabilized)
# + EVI (particle-based Energetic Variational Inference)
# Reports: wall-clock time & MMD^2 vs reference samples from target
# ------------------------------------------------------------
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from numpy.linalg import cholesky, eigh

warnings.filterwarnings("ignore", category=FutureWarning)

# Reproducibility
SEED = 111
np.random.seed(SEED)

# ============================================================
# ============== Utilities (NumPy implementations) ===========
# ============================================================
def logsumexp(a, axis=None):
    m = np.max(a, axis=axis, keepdims=True)
    return np.squeeze(m + np.log(np.sum(np.exp(a - m), axis=axis, keepdims=True)), axis=axis)

def make_spd(S, eps=1e-6):
    evals, evecs = eigh(0.5 * (S + S.T))
    evals = np.clip(evals, eps, None)
    return (evecs * evals) @ evecs.T

def mvn_logpdf(X, mu, Sigma, L=None):
    d = mu.shape[0]
    if L is None:
        L = cholesky(Sigma)
    diff = X - mu
    sol  = np.linalg.solve(L, diff.T)
    quad = np.sum(sol**2, axis=0)
    logdet = 2.0 * np.sum(np.log(np.diag(L)))
    return -0.5 * (quad + d * np.log(2*np.pi) + logdet)

def gmm_logpdf(X, w, mus, Sigmas, chols=None):
    K = len(w)
    if chols is None:
        chols = [cholesky(Sigmas[k]) for k in range(K)]
    parts = np.stack([np.log(w[k]) + mvn_logpdf(X, mus[k], Sigmas[k], chols[k])
                      for k in range(K)], axis=1)
    return logsumexp(parts, axis=1), parts, chols

def sample_gmm(n, w, mus, Sigmas, rng=np.random.default_rng(SEED)):
    K, d = mus.shape
    comps = rng.choice(K, size=n, p=w)
    X = np.empty((n, d))
    for k in range(K):
        idx = np.where(comps == k)[0]
        if idx.size:
            L = cholesky(Sigmas[k])
            eps = rng.normal(size=(idx.size, d))
            X[idx] = mus[k] + eps @ L.T
    return X

# ============================================================
# =================== Star-shaped target =====================
# ============================================================
class StarGaussianNP:
    """Equal-weight K-component mixture on a circle radius r, each arm is skinny."""
    def __init__(self, skewness=100.0, K=5, r=1.5):
        self.K, self.d = K, 2
        theta = 2*np.pi/K
        R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]])
        mu0 = np.array([r, 0.0])
        S0  = np.diag([1.0, 1.0/skewness])
        mus, Sigmas = [], []
        mu, S = mu0.copy(), S0.copy()
        for _ in range(K):
            mus.append(mu.copy())
            Sigmas.append(S.copy())
            mu = R @ mu
            S  = R @ S @ R.T
        self.mus = np.stack(mus, axis=0)
        self.Sigmas = np.stack(Sigmas, axis=0)
        self.ws = np.ones(K) / K
        self._chols = [cholesky(S) for S in self.Sigmas]
        self._invSig = np.stack([np.linalg.inv(S) for S in self.Sigmas], axis=0)

    def logp(self, X):
        parts = []
        for k in range(self.K):
            parts.append(np.log(self.ws[k]) + mvn_logpdf(X, self.mus[k], self.Sigmas[k], self._chols[k]))
        return logsumexp(np.stack(parts, axis=1), axis=1)

    def grad_logp(self, X):
        """∇ log p(X) for the mixture."""
        M, d = X.shape
        logpdfs = np.stack([mvn_logpdf(X, self.mus[k], self.Sigmas[k], self._chols[k])
                            for k in range(self.K)], axis=1)  # (M,K)
        logZ = logsumexp(logpdfs, axis=1)
        R = np.exp(logpdfs - logZ[:, None])  # (M,K)
        grads = np.zeros_like(X)
        for k in range(self.K):
            dx = (self.mus[k] - X)
            grads += (R[:, [k]] * (dx @ self._invSig[k].T))
        return grads

# ============================================================
# ================== EM-GMA (population EM) =================
# ============================================================
def em_gma_population(
    log_pbar, K, d, M_bank=8192, n_iter=80, ridge=1e-5, init_scale=2.0, seed=0
):
    rng = np.random.default_rng(seed)
    w = np.ones(K) / K
    ang = rng.uniform(0, 2*np.pi, size=K)
    mus = np.stack([init_scale*np.cos(ang), init_scale*np.sin(ang)], axis=1)
    Sigmas = np.stack([np.eye(d) for _ in range(K)], axis=0)
    for _ in range(n_iter):
        Z = sample_gmm(M_bank, w, mus, Sigmas, rng=rng)
        log_qz, parts, _ = gmm_logpdf(Z, w, mus, Sigmas)
        log_p  = log_pbar(Z)
        lw = log_p - log_qz
        lw -= lw.max()
        wtil = np.exp(lw)
        omega = wtil / (wtil.sum() + 1e-16)
        R = np.exp(parts - log_qz[:, None])
        Nk = (omega[:, None] * R).sum(axis=0) + 1e-16
        w  = Nk / Nk.sum()
        mus = ((omega[:, None] * R).T @ Z) / Nk[:, None]
        for k in range(K):
            Zc = Z - mus[k]
            Sk = (omega[:, None] * R[:, [k]] * Zc).T @ Zc / Nk[k]
            Sigmas[k] = make_spd(Sk + ridge*np.eye(d))
    return w, mus, Sigmas

# ============================================================
# ============= MMD utilities (evaluation only) =============
# ============================================================
_rng = np.random.default_rng(SEED)
def rbf_K(X, Y, gammas):
    XX = np.sum(X**2, axis=1)[:, None]
    YY = np.sum(Y**2, axis=1)[None, :]
    D2 = XX + YY - 2.0 * (X @ Y.T)
    K = np.zeros_like(D2)
    for g in np.atleast_1d(gammas):
        K += np.exp(-g * D2)
    return K

def median_heuristic_gamma(Z, cap=2000):
    m = min(cap, Z.shape[0])
    idx = _rng.choice(Z.shape[0], size=m, replace=False)
    Zs = Z[idx]
    D2 = np.sum(Zs**2, 1)[:, None] + np.sum(Zs**2, 1)[None, :] - 2.0 * (Zs @ Zs.T)
    tri = np.triu_indices_from(D2, k=1)
    med = np.median(D2[tri]); med = med if med > 1e-12 else 1.0
    return 1.0 / (2.0 * med)

def mmd2_unbiased(X, Y, gammas):
    m, n = X.shape[0], Y.shape[0]
    Kxx = rbf_K(X, X, gammas); np.fill_diagonal(Kxx, 0.0)
    Kyy = rbf_K(Y, Y, gammas); np.fill_diagonal(Kyy, 0.0)
    Kxy = rbf_K(X, Y, gammas)
    return (Kxx.sum() / (m*(m-1))) + (Kyy.sum() / (n*(n-1))) - 2.0 * Kxy.mean()

# ============================================================
# =================== Baseline: MH (NumPy) ===================
# ============================================================
def mh_sampler(logp_np, n_samples, burnin, init, prop_cov):
    d = init.shape[0]
    cur = init.copy()
    cur_lp = logp_np(cur.reshape(1, d))[0]
    samp = []
    rng = np.random.default_rng(SEED+1)
    for t in range(n_samples + burnin):
        prop = cur + rng.multivariate_normal(np.zeros(d), prop_cov)
        prop_lp = logp_np(prop.reshape(1, d))[0]
        if np.log(rng.uniform()) < prop_lp - cur_lp:
            cur, cur_lp = prop, prop_lp
        if t >= burnin:
            samp.append(cur.copy())
    return np.array(samp)

# ============================================================
# ============== Baselines using JAX / Torch / PyMC ==========
# ============================================================
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jnp_logsumexp
import torch
import pymc as pm
import optax

jax_rng = jax.random.PRNGKey(SEED)
torch.manual_seed(SEED)

def star_params_jax(K=5, skewness=100.0, r=1.5):
    theta = 2*np.pi/K
    R = np.array([[np.cos(theta), np.sin(theta)], [-np.sin(theta), np.cos(theta)]], dtype=np.float32)
    mu0 = np.array([r, 0.0], dtype=np.float32)
    S0  = np.diag([1.0, 1.0/skewness]).astype(np.float32)
    mus, Sigmas = [], []
    mu, S = mu0.copy(), S0.copy()
    for _ in range(K):
        mus.append(mu.copy()); Sigmas.append(S.copy())
        mu = R @ mu; S = R @ S @ R.T
    return jnp.array(mus), jnp.array(Sigmas)

mus_jax, Sigmas_jax = star_params_jax()

def mvn_logpdf_jax(z, mu, Sigma):
    diff = z - mu
    sign, logdet = jnp.linalg.slogdet(Sigma)
    sol = jnp.linalg.solve(Sigma, diff)
    return -0.5 * (diff @ sol + 2.0*jnp.log(2*jnp.pi) + logdet)

def log_unnormalized_p_jax(z):
    parts = jax.vmap(lambda mu, S: mvn_logpdf_jax(z, mu, S))(mus_jax, Sigmas_jax)
    return jnp_logsumexp(parts - jnp.log(mus_jax.shape[0]))

mus_t = torch.tensor(np.array(mus_jax), dtype=torch.float32)
Sig_t = torch.tensor(np.array(Sigmas_jax), dtype=torch.float32)
logw_t = torch.log(torch.ones(mus_t.shape[0]) / mus_t.shape[0])

def log_unnormalized_p_torch(z):
    lpks = []
    for k in range(mus_t.shape[0]):
        mvn = torch.distributions.MultivariateNormal(mus_t[k], covariance_matrix=Sig_t[k])
        lpks.append(logw_t[k] + mvn.log_prob(z))
    return torch.logsumexp(torch.stack(lpks), dim=0)

mus_np = np.array(mus_jax); Sig_np = np.array(Sigmas_jax); K_star = mus_np.shape[0]
def log_unnormalized_p_pymc(z):
    parts = []
    for k in range(K_star):
        mv = pm.MvNormal.dist(mu=mus_np[k], cov=Sig_np[k])
        parts.append(pm.logp(mv, z) + np.log(1.0/K_star))
    return pm.math.logsumexp(pm.math.stack(parts))

# ======================= SVGD (JAX) =========================
grad_logp_jax = jax.grad(log_unnormalized_p_jax)

def rbf_kernel_jax(x):
    diffs = x[:, None, :] - x[None, :, :]
    d2 = jnp.sum(diffs**2, axis=-1)
    med2 = jnp.median(d2)
    h = jnp.sqrt(0.5 * med2 / jnp.log(x.shape[0] + 1.0) + 1e-8)
    K = jnp.exp(-d2 / (h**2 + 1e-8))
    gradK = -(2.0 / (h**2 + 1e-8)) * (diffs * K[..., None])
    return K, gradK

@jax.jit
def svgd_step(particles, stepsize):
    score = jax.vmap(grad_logp_jax)(particles)
    K, gradK = rbf_kernel_jax(particles)
    phi = (K @ score + jnp.sum(gradK, axis=1)) / particles.shape[0]
    return particles + stepsize * phi

def run_svgd(init_particles, n_iter=500, stepsize=1e-2):
    x = jnp.array(init_particles)
    for _ in range(n_iter):
        x = svgd_step(x, stepsize)
    return np.array(x)

# ========================= NUTS (PyMC) ======================
def run_nuts(n_samples, warmup, init_pos=None, step_size=0.15):
    with pm.Model() as _model_hmc:
        z = pm.MvNormal("z", mu=np.zeros(2), cov=np.eye(2), shape=2)
        log_prior = pm.logp(pm.MvNormal.dist(mu=np.zeros(2), cov=np.eye(2)), z)
        pm.Potential("tilt", log_unnormalized_p_pymc(z) - log_prior)
        trace = pm.sample(
            draws=n_samples, tune=warmup, chains=1, cores=1,
            step=pm.NUTS(target_accept=0.9),
            random_seed=SEED + 2, progressbar=True, init="jitter+adapt_diag",
        )
    z_samps = trace.posterior["z"].to_numpy().squeeze()
    return z_samps.reshape(-1, z_samps.shape[-1]) if z_samps.ndim == 3 else z_samps

# ====================== Langevin MC (Torch) =================
def langevin_monte_carlo(z0=np.zeros(2), total_steps=4000, lr=1e-2):
    z = torch.tensor(z0, dtype=torch.float32, requires_grad=True)
    out = []
    for _ in range(total_steps):
        logp = log_unnormalized_p_torch(z)
        grad = torch.autograd.grad(logp, z)[0]
        with torch.no_grad():
            z += lr * grad + torch.randn_like(z) * np.sqrt(2 * lr)
        z.requires_grad_(True)
        out.append(z.detach().cpu().numpy().copy())
    return np.array(out)

# ====================== MFVI-ADVI (PyMC) ====================
def run_advi(n_draw=4000, n_fit=20000):
    with pm.Model() as model:
        z = pm.MvNormal("z", mu=np.zeros(2), cov=np.eye(2)*10, shape=2)
        pm.Potential("logp", log_unnormalized_p_pymc(z))
        t0 = time.time(); approx = pm.fit(n=n_fit, method="advi", random_seed=SEED+3)
        advi_time = time.time() - t0
        tr = approx.sample(n_draw, random_seed=SEED+4)
    z_samps = tr.posterior['z'].to_numpy().squeeze()
    return (z_samps.reshape(-1, z_samps.shape[-1]) if z_samps.ndim == 3 else z_samps), advi_time

# ======================= GM-ADVI (JAX) ======================
def run_gmadvi(n_draw=4000, K_mix=40, T_siwae=8, n_steps=3000, lr=1e-2):
    key = jax.random.PRNGKey(SEED+5)
    d = 2
    base = np.array(mus_jax)
    reps = int(np.ceil(K_mix / base.shape[0]))
    means0 = np.vstack([base for _ in range(reps)])[:K_mix]
    means0 = means0 + 0.25 * jax.random.normal(key, (K_mix, d))
    log_scales0 = jnp.log(0.3 + 0.2*jax.random.uniform(key, (K_mix, d)))
    logits0 = jnp.zeros((K_mix,))
    params = {"logits": logits0, "means": means0, "log_scales": log_scales0}

    def scales(p):  # positive with a floor
        return 1e-3 + jax.nn.softplus(p["log_scales"])

    def gmm_log_prob(z, p):
        s = scales(p)
        log_alphas = jax.nn.log_softmax(p['logits'])
        comp_log = -0.5*jnp.sum(((z - p['means'])/s)**2 + 2*jnp.log(s) + jnp.log(2*jnp.pi), axis=-1)
        return jnp_logsumexp(log_alphas + comp_log)

    def siwae_loss(p, key):
        key, sub = jax.random.split(key)
        eps = jax.random.normal(sub, (T_siwae, K_mix, 2))
        z = p['means'][None,:,:] + scales(p)[None,:,:] * eps
        zf = z.reshape(-1, 2)
        lp = jax.vmap(log_unnormalized_p_jax)(zf).reshape(T_siwae, K_mix)
        lq = jax.vmap(gmm_log_prob, in_axes=(0, None))(zf, p).reshape(T_siwae, K_mix)
        log_alphas = jax.nn.log_softmax(p['logits'])
        lw = log_alphas[None,:] + lp - lq
        obj = jax.scipy.special.logsumexp(lw, axis=1) - jnp.log(T_siwae)
        ent_reg = 1e-3 * jnp.sum(jax.nn.softmax(p['logits']) * (jnp.log(jax.nn.softmax(p['logits']) + 1e-12)))
        scale_reg = 1e-4 * jnp.mean(scales(p)**2)
        return -jnp.mean(obj) + ent_reg + scale_reg

    opt = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(lr))
    opt_state = opt.init(params)

    @jax.jit
    def update(p, opt_state, key):
        loss, grads = jax.value_and_grad(siwae_loss)(p, key)
        updates, opt_state = opt.update(grads, opt_state)
        p = optax.apply_updates(p, updates)
        return p, opt_state, loss

    t0 = time.time()
    for _ in range(n_steps):
        key, sub = jax.random.split(key)
        params, opt_state, loss = update(params, opt_state, sub)
    gmadvi_time = time.time() - t0

    # Draw exactly n_draw samples
    key, ckey, nkey = jax.random.split(key, 3)
    alphas = jax.nn.softmax(params['logits'])
    comps = jax.random.choice(ckey, K_mix, (n_draw,), p=alphas)
    z = jax.random.normal(nkey, (n_draw, 2)) * scales(params)[comps] + params['means'][comps]
    z = np.array(z)
    assert np.isfinite(z).all(), "GM-ADVI produced non-finite samples; try lowering lr or steps."
    return z, gmadvi_time

# ============================================================
# ============= EVI (Energetic Variational) =================
# ============================================================
class EVI:
    def _rbf_kernel(self, x, h=None):
        diff = x[:, None, :] - x[None, :, :]
        d2 = np.sum(diff**2, axis=-1)
        if h is None:
            med2 = np.median(d2)
            h = np.sqrt(0.5 * med2 / np.log(x.shape[0] + 1.0) + 1e-12)
        kxy = np.exp(-d2 / (2.0*(h**2) + 1e-12))
        sumkxy = np.sum(kxy, axis=1, keepdims=True)
        gradK = -diff * kxy[:, :, None] / (h**2 + 1e-12)
        dxkxy = np.sum(gradK, axis=0)
        obj = np.sum(np.transpose(gradK, (1,0,2)) / np.clip(sumkxy, 1e-12, None), axis=1)
        return dxkxy, sumkxy, obj

    def _vector_field(self, x, x0, grad, tau, h=None):
        dxkxy, sumkxy, obj = self._rbf_kernel(x, h=h)
        return (x - x0)/tau + ( - dxkxy/np.clip(sumkxy,1e-12,None) - obj - grad )

    def run(self, x0, grad_log_p_fn, inner_iter=20, outer_iter=5, tau=1e-1, lr=0.1, h=None):
        x = x0.copy()
        x_initial = x0.copy()
        for _ in range(outer_iter):
            adag = np.zeros_like(x)
            for _ in range(inner_iter):
                grad = grad_log_p_fn(x)
                v = self._vector_field(x, x_initial, grad, tau, h=h)
                adag += v**2
                x = x - lr * v / np.sqrt(adag + 1e-12)
            x_initial = x
        return x

def run_evi(star_obj, n_particles, inner_iter=20, outer_iter=5, tau=1e-1, lr=0.1, h=None, seed=SEED+30):
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=(n_particles, 2))
    t0 = time.time()
    solver = EVI()
    X = solver.run(x0, grad_log_p_fn=star_obj.grad_logp,
                   inner_iter=inner_iter, outer_iter=outer_iter, tau=tau, lr=lr, h=h)
    evi_time = time.time() - t0
    return X, evi_time

# ============================================================
# ======================= main ===============================
# ============================================================
# ----- target -----
star = StarGaussianNP(skewness=100.0, K=5, r=1.5)
def log_pbar_np(Z): return star.logp(Z)

# Reference samples (for MMD evaluation ONLY)
N_REF = 2000
X_ref = sample_gmm(N_REF, star.ws, star.mus, star.Sigmas)

# Bandwidths for MMD
gamma0 = median_heuristic_gamma(X_ref)
gammas = gamma0 * np.array([0.5, 1.0, 2.0])

# EXACT number of samples for every method
N_DRAW = 2000

results = {}

# 1) EM-GMA (data-free)
print("\n=== EM-GMA (data-free) ===")
t0 = time.time()
w_gma, mu_gma, Sig_gma = em_gma_population(
    log_pbar=log_pbar_np, K=5, d=2, M_bank=8192, n_iter=80, ridge=1e-5, init_scale=2.0, seed=SEED+10
)
emgma_time = time.time() - t0
X_emgma = sample_gmm(N_DRAW, w_gma, mu_gma, Sig_gma)
results["EM-GMA"] = {"time": emgma_time, "mmd2": mmd2_unbiased(X_ref, X_emgma, gammas)}

# 2) Metropolis-Hastings
print("\n=== Metropolis-Hastings ===")
t0 = time.time()
X_mh = mh_sampler(star.logp, n_samples=N_DRAW, burnin=1000, init=np.zeros(2), prop_cov=0.2*np.eye(2))
mh_time = time.time() - t0
results["MH"] = {"time": mh_time, "mmd2": mmd2_unbiased(X_ref, X_mh, gammas)}

# 3) NUTS (PyMC)
print("\n=== NUTS (HMC) ===")
t0 = time.time()
X_nuts = run_nuts(n_samples=N_DRAW, warmup=1000)
nuts_time = time.time() - t0
results["NUTS (HMC)"] = {"time": nuts_time, "mmd2": mmd2_unbiased(X_ref, X_nuts, gammas)}

# 4) LMC (PyTorch)
print("\n=== Langevin Monte Carlo ===")
t0 = time.time()
X_lmc = langevin_monte_carlo(z0=np.zeros(2), total_steps=N_DRAW, lr=1e-2)
lmc_time = time.time() - t0
results["LMC"] = {"time": lmc_time, "mmd2": mmd2_unbiased(X_ref, X_lmc, gammas)}

# 5) SVGD (JAX)
print("\n=== SVGD ===")
rng = np.random.default_rng(SEED+20)
init_svgd = rng.normal(size=(N_DRAW, 2))
t0 = time.time()
X_svgd = run_svgd(init_svgd, n_iter=1000, stepsize=1e-2)
svgd_time = time.time() - t0
results["SVGD"] = {"time": svgd_time, "mmd2": mmd2_unbiased(X_ref, X_svgd, gammas)}

# 6) MFVI-ADVI (PyMC)
print("\n=== MFVI-ADVI (PyMC) ===")
X_advi, advi_time = run_advi(n_draw=N_DRAW, n_fit=20000)
results["MFVI-ADVI"] = {"time": advi_time, "mmd2": mmd2_unbiased(X_ref, X_advi, gammas)}

# 7) GM-ADVI (JAX, stabilized)
print("\n=== GM-ADVI (JAX, stabilized) ===")
X_gmadvi, gmadvi_time = run_gmadvi(n_draw=N_DRAW, K_mix=40, T_siwae=8, n_steps=3000, lr=1e-2)
results["GM-ADVI"] = {"time": gmadvi_time, "mmd2": mmd2_unbiased(X_ref, X_gmadvi, gammas)}

# 8) EVI (particle based) — EXACTLY N_DRAW particles
print("\n=== EVI (particle EVI) ===")
X_evi, evi_time = run_evi(star, n_particles=N_DRAW, inner_iter=20, outer_iter=5, tau=1e-1, lr=0.1, h=None)
results["EVI"] = {"time": evi_time, "mmd2": mmd2_unbiased(X_ref, X_evi, gammas)}

# ==================== Report ====================
order = ["EM-GMA","MH","NUTS (HMC)","LMC","SVGD","MFVI-ADVI","GM-ADVI","EVI"]
print("\n=== Execution time (s) and MMD^2 (↓ is better) ===")
print("{:<12s}  {:>12s}  {:>12s}".format("Method","Time (s)","MMD^2"))
for m in order:
    print("{:<12s}  {:>12.3f}  {:>12.4e}".format(m, results[m]["time"], results[m]["mmd2"]))

# ==================== Comparison grid (2x4) ====================
xs = np.linspace(-4, 4, 320)
ys = np.linspace(-4, 4, 320)
Xg, Yg = np.meshgrid(xs, ys)
grid = np.stack([Xg.ravel(), Yg.ravel()], axis=1)
Zlog = star.logp(grid).reshape(Xg.shape)

all_samples = {
    "EM-GMA": X_emgma,
    "MH": X_mh,
    "NUTS (HMC)": X_nuts,
    "LMC": X_lmc,
    "SVGD": X_svgd,
    "MFVI-ADVI": X_advi,
    "GM-ADVI": X_gmadvi,
    "EVI": X_evi,
}

fig, axes = plt.subplots(2, 4, figsize=(18, 8), sharex=True, sharey=True)
axes = axes.ravel()
for i, m in enumerate(order):
    ax = axes[i]
    ax.contour(Xg, Yg, Zlog, levels=18, colors='k', alpha=0.25, linewidths=0.7, linestyles='dotted')
    ax.scatter(X_ref[:, 0], X_ref[:, 1], s=5, alpha=0.15, color='tab:orange', label='reference')
    S = all_samples[m]
    ax.scatter(S[:, 0], S[:, 1], s=6, alpha=0.45, color='tab:blue', label=m)
    ax.set_title(f"{m}\nTime: {results[m]['time']:.2f}s | MMD$^2$: {results[m]['mmd2']:.2e}", fontsize=11)
    ax.set_xlim([-4, 4]); ax.set_ylim([-4, 4])
    if i % 4 == 0: ax.set_ylabel('$x_2$')
    if i >= 4: ax.set_xlabel('$x_1$')

plt.tight_layout()
plt.show()

"""# end."""
