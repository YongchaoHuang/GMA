# -*- coding: utf-8 -*-
"""[used] GMA sampling test 18: sensor localisation.ipynb

#yongchao.huang@abdn.ac.uk
"""

!lscpu

import psutil

print("CPU cores:", psutil.cpu_count(logical=False))
print("Logical CPUs:", psutil.cpu_count(logical=True))
print("Memory (GB):", round(psutil.virtual_memory().total / 1e9, 2))
print("Disk space (GB):", round(psutil.disk_usage('/').total / 1e9, 2))

"""# Begin.

# 5 known, 6 unknown sensors.

## use ground truth as reference for computing REM, rather than HMC mean.
"""

# sensor_localization_vi_all_with_LMA_5anchors.py
# ------------------------------------------------------------
# Sensor Network Localization (Ahn et al., 2013; Guo et al., 2017 BVI)
# Methods: HMC/NUTS, MFVI-ADVI, GM-ADVI, S-ADVI, BVI, LMA, EM-GMA
# Parameterization in u-space (R^D) with x = sigmoid(u) ∈ (0,1)^D.
# For PyMC methods we add the correct log-uniform prior in u via a Potential.
# Configuration in this script: N=11, N_ANCH=5 (known), N_UNK=6 (unknown), DIMS=12.
# ------------------------------------------------------------
import time, math, warnings
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)

# Reproducibility
SEED = 123
rng_np = np.random.default_rng(SEED)

# ===================== Problem constants ====================
N = 11                    # total sensors
N_ANCH = 5                # number of known sensors (anchors)
N_UNK = N - N_ANCH        # 6 unknown sensors
DIMS = 2 * N_UNK          # infer 12 unknown coordinates

# Geometric / observation settings (Ahn+2013 / Guo+2017)
LIM = 1.2                 # square region side (placements in [0, 1.2]^2)
R = 0.3                   # link-length scale in p(Z=1 | d) = exp(-d^2/(2 R^2))
SIGMA = 0.02              # range noise std for observed links

# Five anchors (well-spread)
anchors = np.array([
    [0.08, 0.08],
    [1.12, 0.08],
    [0.08, 1.12],
    [1.12, 1.12],
    [0.60, 0.60],
], dtype=np.float32)

# Ground-truth unknowns (uniform in [0,1.2]^2)
theta_true = np.zeros((N, 2), dtype=np.float32)
theta_true[:N_ANCH] = anchors
theta_true[N_ANCH:] = rng_np.uniform(0.05, 1.15, size=(N_UNK, 2)).astype(np.float32)

# Pair indices (i<j)
pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
P = len(pairs)

def pairwise_dist(Theta):  # Theta shape (N,2)
    D = np.zeros((N, N), dtype=np.float32)
    for (i, j) in pairs:
        d = float(np.linalg.norm(Theta[i] - Theta[j]))
        D[i, j] = D[j, i] = d
    return D

# Generate synthetic Z,Y (choose top-14 most likely links, as in the papers)
D_true = pairwise_dist(theta_true)
Pi = np.exp(- (D_true**2) / (2.0 * R**2))

P_obs = 14
flat_probs = np.array([Pi[i, j] for (i, j) in pairs], dtype=np.float32)
keep = np.argsort(-flat_probs)[:P_obs]
keep_mask = np.zeros(P, dtype=bool); keep_mask[keep] = True

Z = np.zeros((N, N), dtype=int)
Y = np.zeros((N, N), dtype=np.float32)
for k, (i, j) in enumerate(pairs):
    if keep_mask[k]:
        Z[i, j] = Z[j, i] = 1
        yij = rng_np.normal(D_true[i, j], SIGMA)
        Y[i, j] = Y[j, i] = max(0.0, yij)
    else:
        Z[i, j] = Z[j, i] = 0
        Y[i, j] = Y[j, i] = 0.0

obs_pairs = [(i, j) for (i, j) in pairs if Z[i, j] == 1]
miss_pairs = [(i, j) for (i, j) in pairs if Z[i, j] == 0]
print(f"Data: N={N} (anchors={N_ANCH}, unknown={N_UNK}), |obs|={len(obs_pairs)}, |miss|={len(miss_pairs)}")

# ------------------- helpers: x/u packing & sigmoid -------------------
def unpack_x(x_vec):
    out = np.zeros((N, 2), dtype=np.float32)
    out[:N_ANCH] = anchors
    out[N_ANCH:] = x_vec.reshape(N_UNK, 2)
    return out

def sigmoid_np(u):
    return 1.0 / (1.0 + np.exp(-u))

# ================== Likelihood in x-space (NumPy) =====================
def loglik_x_numpy(Theta):
    ll = 0.0
    for (i, j) in obs_pairs:
        d = np.linalg.norm(Theta[i] - Theta[j]) + 1e-12
        p1 = math.exp(- (d*d) / (2.0 * R*R))
        ll += math.log(max(p1, 1e-300))
        ll += -0.5*((Y[i, j] - d)**2)/(SIGMA**2) - math.log(SIGMA*math.sqrt(2.0*math.pi))
    for (i, j) in miss_pairs:
        d = np.linalg.norm(Theta[i] - Theta[j]) + 1e-12
        p1 = math.exp(- (d*d) / (2.0 * R*R))
        ll += math.log(max(1.0 - p1, 1e-300))
    return ll

# ====================== PyMC (PyTensor) pieces ========================
import pymc as pm
import pytensor.tensor as at

def at_sigmoid(x):
    """Numerically stable sigmoid using core PyTensor ops (no nnet)."""
    return at.where(
        at.ge(x, 0),
        1.0 / (1.0 + at.exp(-x)),
        at.exp(x) / (1.0 + at.exp(x)),
    )

def sensor_loglike_at_u(u_vec_at):
    """
    PyTensor log-likelihood as a function of unconstrained u (DIMS,).
    We map x = sigmoid(u) ∈ (0,1) and build full Θ = [anchors; x].
    """
    x = at_sigmoid(u_vec_at)                      # (DIMS,)
    x2 = x.reshape((N_UNK, 2))
    anchors_const = at.as_tensor_variable(anchors.astype(np.float32))
    Theta = at.concatenate([anchors_const, x2], axis=0)  # (N,2)

    # vectorize across pairs
    idx_i = np.array([i for (i, j) in pairs], dtype="int64")
    idx_j = np.array([j for (i, j) in pairs], dtype="int64")
    Theta_i = Theta[idx_i]; Theta_j = Theta[idx_j]
    diff = Theta_i - Theta_j
    d = at.sqrt(at.sum(diff**2, axis=1) + 1e-12)
    p1 = at.exp(-(d**2) / (2.0 * (R**2)))

    obs_mask = np.array([Z[i, j] == 1 for (i, j) in pairs], dtype=bool)
    mis_mask = ~obs_mask

    d_obs = d[obs_mask]
    p1_obs = p1[obs_mask]
    y_obs_np = np.array([Y[i, j] for (i, j) in pairs if Z[i, j] == 1], dtype=np.float32)
    y_obs = at.as_tensor_variable(y_obs_np)

    const_norm = float(np.log(SIGMA * np.sqrt(2.0 * np.pi)))
    ll_obs = at.sum(
        at.log(at.clip(p1_obs, 1e-300, 1e300))
        - 0.5 * ((y_obs - d_obs) ** 2) / (SIGMA**2)
        - const_norm
    )
    p1_mis = p1[mis_mask]
    ll_mis = at.sum(at.log(at.clip(1.0 - p1_mis, 1e-300, 1e300)))
    return ll_obs + ll_mis

def log_uniform_prior_in_u_at(u):
    """
    If x ~ Uniform(0,1) and x = sigmoid(u), then p(u) ∝ sigmoid(u)(1-sigmoid(u)).
    So log p(u) = sum_i [ log σ(u_i) + log(1-σ(u_i)) ] + const.
    """
    sx = at_sigmoid(u)
    return at.sum(at.log(sx + 1e-12) + at.log(1.0 - sx + 1e-12))

# HMC/NUTS in u-space with correct tilt
def run_hmc(draws=2500, tune=1000):
    with pm.Model() as model:
        u = pm.Normal("u", mu=0.0, sigma=10.0, shape=DIMS)   # base prior in u
        base_logp = pm.logp(pm.Normal.dist(0.0, 10.0), u)    # to remove base prior
        tilt = sensor_loglike_at_u(u) + log_uniform_prior_in_u_at(u) - base_logp
        pm.Potential("tilt", tilt)
        trace = pm.sample(
            draws=draws, tune=tune, chains=1, cores=1,
            step=pm.NUTS(target_accept=0.9), init="jitter+adapt_diag",
            random_seed=SEED, progressbar=True
        )
    U = trace.posterior["u"].to_numpy().squeeze()
    return U.reshape(-1, DIMS) if U.ndim == 3 else U

def run_advi(n_fit=20000, n_draw=4000):
    with pm.Model() as model:
        u = pm.Normal("u", mu=0.0, sigma=10.0, shape=DIMS)
        base_logp = pm.logp(pm.Normal.dist(0.0, 10.0), u)
        pm.Potential("tilt", sensor_loglike_at_u(u) + log_uniform_prior_in_u_at(u) - base_logp)
        t0 = time.time()
        approx = pm.fit(n=n_fit, method="advi", random_seed=SEED+1)
        spent = time.time() - t0
        tr = approx.sample(n_draw, random_seed=SEED+2)
    U = tr.posterior["u"].to_numpy().squeeze()
    U = U.reshape(-1, DIMS) if U.ndim == 3 else U
    return U, spent

# =============================== JAX setup ===============================
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jls
import optax

key = jax.random.PRNGKey(SEED)
pairs_arr = jnp.array(pairs, dtype=jnp.int32)
obs_arr = jnp.array(obs_pairs, dtype=jnp.int32)
mis_arr = jnp.array(miss_pairs, dtype=jnp.int32)
anchors_j = jnp.array(anchors, dtype=jnp.float32)
Y_j = jnp.array(Y, dtype=jnp.float32)
R_J = jnp.float32(R)
SIGMA_J = jnp.float32(SIGMA)

def sigmoid_j(x):
    return jnp.where(x >= 0, 1/(1+jnp.exp(-x)), jnp.exp(x)/(1+jnp.exp(x)))

def unpack_u_to_full_x(u_vec):
    x = sigmoid_j(u_vec).reshape(N_UNK, 2)
    return jnp.concatenate([anchors_j, x], axis=0)

def loglik_u_j(u_vec):
    Theta = unpack_u_to_full_x(u_vec)
    def d_ij(idx):
        i,j = idx[0], idx[1]
        return jnp.linalg.norm(Theta[i]-Theta[j]) + 1e-12
    d_obs = jax.vmap(d_ij)(obs_arr)
    p1_obs = jnp.exp(- (d_obs**2) / (2.0 * R_J**2))
    y_obs = Y_j[tuple(obs_arr.T)]
    ll_obs = jnp.sum(jnp.log(jnp.clip(p1_obs, 1e-300, None))
                     - 0.5*((y_obs - d_obs)**2)/(SIGMA_J**2)
                     - jnp.log(SIGMA_J*jnp.sqrt(2*jnp.pi)))
    d_mis = jax.vmap(d_ij)(mis_arr)
    p1_mis = jnp.exp(- (d_mis**2) / (2.0 * R_J**2))
    ll_mis = jnp.sum(jnp.log(jnp.clip(1.0 - p1_mis, 1e-300, None)))
    return ll_obs + ll_mis

def logprior_u_j(u_vec):
    s = sigmoid_j(u_vec)
    return jnp.sum(jnp.log(s + 1e-12) + jnp.log(1.0 - s + 1e-12))

def logf_u_j(u_vec):
    return loglik_u_j(u_vec) + logprior_u_j(u_vec)

grad_logf_u = jax.grad(logf_u_j)

# ===================== GM-ADVI (stabilized, u-space) =====================
def run_gmadvi(n_steps=3000, n_comp=24, T=8, lr=5e-3, n_draw=4000):
    key = jax.random.PRNGKey(SEED+10)
    d = DIMS
    means0 = 0.0 + 0.25*jax.random.normal(key, (n_comp, d))
    log_scales0 = jnp.log(0.2 + 0.2*jax.random.uniform(key, (n_comp, d)))
    logits0 = jnp.zeros((n_comp,))
    params = {"logits": logits0, "means": means0, "log_scales": log_scales0}
    def scales(p): return 1e-3 + jax.nn.softplus(p["log_scales"])
    def logq(z, p):
        s = scales(p)
        logw = jax.nn.log_softmax(p["logits"])
        comp = -0.5*jnp.sum(((z - p["means"])/s)**2 + 2*jnp.log(s) + jnp.log(2*jnp.pi), axis=-1)
        return jls(logw + comp, axis=-1)
    def siwae_loss(p, key):
        key, sub = jax.random.split(key)
        eps = jax.random.normal(sub, (T, n_comp, d))
        z = p["means"][None,:,:] + scales(p)[None,:,:]*eps
        zf = z.reshape(-1, d)
        lp = jax.vmap(logf_u_j)(zf).reshape(T, n_comp)
        lq = jax.vmap(lambda row: logq(row, p))(zf).reshape(T, n_comp)
        lw = jax.nn.log_softmax(p["logits"])[None,:] + lp - lq
        obj = jax.scipy.special.logsumexp(lw, axis=1) - jnp.log(T)
        ent_reg = 1e-3*jnp.sum(jax.nn.softmax(p["logits"]) * (jnp.log(jax.nn.softmax(p["logits"])+1e-12)))
        scale_reg = 1e-4*jnp.mean(scales(p)**2)
        return -jnp.mean(obj) + ent_reg + scale_reg
    opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr))
    state = opt.init(params)
    @jax.jit
    def step(p, st, key):
        loss, g = jax.value_and_grad(siwae_loss)(p, key)
        upd, st = opt.update(g, st)
        p = optax.apply_updates(p, upd)
        return p, st, loss
    t0 = time.time()
    for _ in range(n_steps):
        key, sub = jax.random.split(key)
        params, state, _ = step(params, state, sub)
    spent = time.time()-t0
    key, ckey, nkey = jax.random.split(key, 3)
    alphas = jax.nn.softmax(params["logits"])
    comps = jax.random.choice(ckey, n_comp, (n_draw,), p=alphas)
    z = jax.random.normal(nkey, (n_draw, d))* (1e-3 + jax.nn.softplus(params["log_scales"]))[comps] + params["means"][comps]
    return np.array(z), spent

# ===================== S-ADVI (monotone warp, u-space) =====================
def run_sadvi(n_steps=4000, lr=1e-2, K=4, n_draw=4000):
    key = jax.random.PRNGKey(SEED+20)
    d = DIMS
    params = {
        "mu": jnp.zeros((d,)),
        "log_sigma": jnp.log(jnp.full((d,), 0.7)),
        "a": jnp.log(jnp.ones((d,K))*0.1),
        "b": jnp.log(jnp.ones((d,K))*0.5),
        "c": jnp.zeros((d,K))
    }
    def transform(eps, p):
        sig = 1e-3 + jax.nn.softplus(p["log_sigma"])
        A = jax.nn.softplus(p["a"]); B = jax.nn.softplus(p["b"]); C = p["c"]
        S = jax.nn.sigmoid(B*eps[...,None] + C)          # (..., d, K)
        u = p["mu"] + sig*( eps + jnp.sum(A*S, axis=-1) )
        du_de = sig*( 1.0 + jnp.sum(A*B*S*(1.0-S), axis=-1) )
        return u, du_de
    def elbo(p, key, T=256):
        key, sub = jax.random.split(key)
        eps = jax.random.normal(sub, (T, d))
        u, jac = transform(eps, p)
        logq = -0.5*jnp.sum(eps**2, axis=-1) - 0.5*d*jnp.log(2*jnp.pi) - jnp.sum(jnp.log(jac), axis=-1)
        lp = jax.vmap(logf_u_j)(u)
        return jnp.mean(lp - logq)
    opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr))
    state = opt.init(params)
    @jax.jit
    def step(p, st, key):
        loss, g = jax.value_and_grad(lambda pr: -elbo(pr, key))(p)
        upd, st = opt.update(g, st)
        p = optax.apply_updates(p, upd)
        return p, st, loss
    t0 = time.time()
    for _ in range(n_steps):
        key, sub = jax.random.split(key)
        params, state, _ = step(params, state, sub)
    spent = time.time()-t0
    key, sub = jax.random.split(key)
    eps = jax.random.normal(sub, (n_draw, d))
    u, _ = transform(eps, params)
    return np.array(u), spent

# ===================== SPD utilities (shared) =====================
def nearest_spd(A: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    """Project a symmetric matrix to the nearest SPD by eigvalue flooring."""
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, floor)
    return (V * w) @ V.T

def safe_cholesky(S: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """Try Cholesky; if it fails, project to SPD and retry."""
    try:
        return np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        S2 = nearest_spd(S, floor=floor)
        return np.linalg.cholesky(S2)

# ===================== BVI (Laplace-boosted mixture, u-space) =====================
class BVIMix:
    def __init__(self, d, init_scale=1.0):
        self.w = [1.0]
        self.means = [np.zeros((d,), dtype=float)]
        self.covs = [np.eye(d, dtype=float) * (init_scale**2)]
        self.d = d
    def logq(self, z):
        logws = jnp.log(jnp.array(self.w))
        comp = []
        for m, S in zip(self.means, self.covs):
            S_j = jnp.array(S)
            diff = z - jnp.array(m)
            sign, logdet = jnp.linalg.slogdet(S_j)
            sol = jnp.linalg.solve(S_j, diff)
            comp.append(-0.5*jnp.sum(diff*sol, axis=-1) - 0.5*self.d*jnp.log(2*jnp.pi) - 0.5*logdet)
        comp = jnp.stack(comp, axis=-1)
        return jls(logws + comp, axis=-1)
    def add_component(self, mean: np.ndarray, cov: np.ndarray, alpha: float, floor: float = 1e-6):
        alpha = float(alpha)
        self.w = [(1-alpha)*w for w in self.w] + [alpha]
        self.means.append(np.array(mean, dtype=float))
        self.covs.append(nearest_spd(np.array(cov, dtype=float), floor=floor))  # SPD guard

def run_bvi(T=12, iters_mode=250, lr=5e-2, alpha_grid=None, n_draw=4000, hess_floor=1e-5):
    if alpha_grid is None:
        alpha_grid = np.linspace(0.05, 0.5, 10)
    d = DIMS
    mix = BVIMix(d, init_scale=0.8)
    t0 = time.time()
    for t in range(T):
        # maximize stabilized residual r(v) = log f(v) - log q_{t-1}(v)
        def r(v):
            return logf_u_j(v) - mix.logq(v)
        v = 0.0 + 0.2*jax.random.normal(jax.random.PRNGKey(SEED+100+t), (d,))
        opt = optax.adam(lr)
        st = opt.init(v)
        @jax.jit
        def step(v, st):
            loss, g = jax.value_and_grad(lambda x: -(r(x)))(v)
            upd, st = opt.update(g, st)
            v = optax.apply_updates(v, upd)
            return v, st, loss
        for _ in range(iters_mode):
            v, st, _ = step(v, st)

        # Laplace Hessian (neg. Hessian of residual)
        H = jax.hessian(lambda z: -(logf_u_j(z) - mix.logq(z)))(v)
        H = np.array(H, dtype=float)
        Prec = nearest_spd(H, floor=hess_floor)     # ensure positive definite precision
        cov = np.linalg.inv(Prec)
        cov = nearest_spd(cov, floor=hess_floor)    # ensure SPD covariance

        # draw zs in NumPy, then convert to jnp once
        L = safe_cholesky(cov, floor=hess_floor)
        zs_np = np.array(v) + rng_np.normal(size=(1024, d)) @ L.T
        zs = jnp.array(zs_np)
        cov_j = jnp.array(cov)
        v_j = jnp.array(v)

        # choose alpha by MC ELBO in pure JAX
        def elbo_alpha(a_float):
            a = jnp.float32(a_float)
            log1ma = jnp.log(1.0 - a + 1e-12)
            loga   = jnp.log(a + 1e-12)
            logmixw = jnp.stack([log1ma, loga])  # shape (2,)

            def logq_new(z):
                diff = z - v_j
                sign, logdet = jnp.linalg.slogdet(cov_j)
                comp_new = -0.5*jnp.sum(diff * jnp.linalg.solve(cov_j, diff), axis=-1) \
                           - 0.5*d*jnp.log(2*jnp.pi) - 0.5*logdet
                # mixture of (old q, new Gaussian at v)
                two_terms = jnp.stack([mix.logq(z), comp_new], axis=-1)  # (..., 2)
                return jls(logmixw + two_terms, axis=-1)

            return jnp.mean(jax.vmap(logf_u_j)(zs) - jax.vmap(logq_new)(zs))

        scores = np.array([float(elbo_alpha(a)) for a in alpha_grid])
        a_best = float(alpha_grid[int(np.argmax(scores))])
        mix.add_component(np.array(v), cov, a_best, floor=hess_floor)   # store SPD covariance

    spent = time.time()-t0
    # sample
    w = np.array(mix.w)/np.sum(mix.w)
    comps = rng_np.choice(len(w), size=n_draw, p=w)
    z = np.zeros((n_draw, d))
    for k in range(len(w)):
        idx = np.where(comps==k)[0]
        if idx.size:
            Lc = safe_cholesky(mix.covs[k], floor=hess_floor)  # robust Cholesky
            eps = rng_np.normal(size=(idx.size, d))
            z[idx] = mix.means[k] + eps @ Lc.T
    return z, spent

# ===================== LMA (Laplace Mixture Approx.) =====================
def lma_find_modes(n_starts=16, iters=600, lr=5e-2, tol_merge=1e-2):
    """Multi-start gradient ascent on log f(u) to discover distinct modes."""
    modes = []
    vals = []
    opt = optax.chain(optax.clip_by_global_norm(20.0), optax.adam(lr))
    for s in range(n_starts):
        key = jax.random.PRNGKey(SEED + 300 + s)
        u = 0.0 + 0.5 * jax.random.normal(key, (DIMS,))
        state = opt.init(u)
        @jax.jit
        def step(u, st):
            loss, g = jax.value_and_grad(lambda v: -logf_u_j(v))(u)
            updates, st = opt.update(g, st)
            u = optax.apply_updates(u, updates)
            return u, st, loss
        for _ in range(iters):
            u, state, _ = step(u, state)
        u_np = np.array(u)
        val = float(logf_u_j(u_np))
        # merge near-duplicates (in x-space distance)
        keep = True
        x_u = sigmoid_np(u_np)
        for m in modes:
            if np.linalg.norm(sigmoid_np(m) - x_u) < tol_merge:
                keep = False; break
        if keep:
            modes.append(u_np); vals.append(val)
    idx = np.argsort(-np.array(vals))
    modes = [modes[i] for i in idx]
    vals  = [vals[i]  for i in idx]
    return modes, vals

def lma_build_mixture(modes, vals, kappa=1.0, floor=1e-6):
    """At each mode, Σ = kappa^2 * inv(-∇² log f). Weights via Laplace evidence."""
    comps = []
    d = DIMS
    log_evid = []
    for u_mode, val in zip(modes, vals):
        H = np.array(jax.hessian(logf_u_j)(u_mode), dtype=float)  # Hessian at mode
        Prec = nearest_spd(-H, floor=floor)                       # positive definite
        Sigma = np.linalg.inv(Prec)
        Sigma = nearest_spd((kappa**2) * Sigma, floor=floor)
        sign, logdet = np.linalg.slogdet(Sigma)
        log_evid.append(val + 0.5*d*math.log(2*math.pi) + 0.5*logdet)
        comps.append({"mu": np.array(u_mode, dtype=float), "Sigma": Sigma})
    log_evid = np.array(log_evid); m = log_evid.max()
    w = np.exp(log_evid - m); w /= w.sum()
    for c, wi in zip(comps, w): c["w"] = float(wi)
    return comps

def lma_sample(comps, n_draw=4000, rng=rng_np, floor=1e-8):
    w = np.array([c["w"] for c in comps], dtype=float)
    J = rng.choice(len(comps), size=n_draw, p=w)
    U = np.zeros((n_draw, DIMS), dtype=float)
    for k in range(len(comps)):
        idx = np.where(J==k)[0]
        if idx.size:
            L = safe_cholesky(comps[k]["Sigma"], floor=floor)
            eps = rng.normal(size=(idx.size, DIMS))
            U[idx] = comps[k]["mu"] + eps @ L.T
    return U

def run_lma(n_starts=16, iters=600, lr=5e-2, kappa=1.0, n_draw=4000):
    t0 = time.time()
    modes, vals = lma_find_modes(n_starts=n_starts, iters=iters, lr=lr)
    comps = lma_build_mixture(modes, vals, kappa=kappa)
    U = lma_sample(comps, n_draw=n_draw, rng=rng_np)
    spent = time.time() - t0
    return U, spent

# ===================== EM-GMA (population EM; SNIS, u-space) =====================
def make_spd(S, eps=1e-6):
    evals, evecs = np.linalg.eigh(0.5*(S + S.T))
    evals = np.clip(evals, eps, None)
    return (evecs * evals) @ evecs.T

def gmm_logpdf(X, w, mus, Sigmas, chols=None):
    K = len(w); d = X.shape[1]
    if chols is None:
        chols = [np.linalg.cholesky(Sigmas[k]) for k in range(K)]
    parts = np.stack([
        np.log(w[k]) - 0.5*np.sum((np.linalg.solve(chols[k], (X - mus[k]).T))**2, axis=0)
        - 0.5*d*np.log(2*np.pi) - np.log(np.diag(chols[k])).sum()
        for k in range(K)
    ], axis=1)
    m = np.max(parts, axis=1, keepdims=True)
    log_qz = (m + np.log(np.sum(np.exp(parts - m), axis=1, keepdims=True))).squeeze(1)
    return log_qz, parts, chols

def sample_gmm(n, w, mus, Sigmas, rng=rng_np):
    K, d = mus.shape
    comps = rng.choice(K, size=n, p=w)
    X = np.empty((n, d))
    for k in range(K):
        idx = np.where(comps==k)[0]
        if idx.size:
            L = np.linalg.cholesky(Sigmas[k])
            eps = rng.normal(size=(idx.size, d))
            X[idx] = mus[k] + eps @ L.T
    return X

def logf_u_numpy(u_vec):
    x = sigmoid_np(u_vec)
    Theta = unpack_x(x)
    return loglik_x_numpy(Theta) + np.sum(np.log(x + 1e-12) + np.log(1.0 - x + 1e-12))

def em_gma_population(log_f, K=24, M_bank=8192, n_iter=60, ridge=1e-5, init_scale=0.9, seed=SEED+40):
    rng_local = np.random.default_rng(seed)
    w = np.ones(K)/K
    mus = 0.0 + init_scale*rng_local.normal(size=(K, DIMS))
    Sigmas = np.stack([np.eye(DIMS) for _ in range(K)], axis=0)
    for _ in range(n_iter):
        Zb = sample_gmm(M_bank, w, mus, Sigmas, rng_local)
        log_qz, parts, _ = gmm_logpdf(Zb, w, mus, Sigmas)
        log_p  = np.array([log_f(z) for z in Zb])
        lw = log_p - log_qz
        lw -= lw.max()
        omega = np.exp(lw); omega /= (omega.sum() + 1e-16)
        Rm = np.exp(parts - log_qz[:,None])
        Nk = (omega[:,None] * Rm).sum(axis=0) + 1e-16
        w  = Nk / Nk.sum()
        mus = ((omega[:,None] * Rm).T @ Zb) / Nk[:,None]
        for k in range(K):
            Zc = Zb - mus[k]
            Sk = (omega[:,None] * Rm[:,[k]] * Zc).T @ Zc / Nk[k]
            Sigmas[k] = make_spd(Sk + ridge*np.eye(DIMS))
    return w, mus, Sigmas

# ===================== Utilities: REM, plotting =====================
def REM_x(est_samps_x, ref_mean_x):
    m = est_samps_x.mean(axis=0)
    num = np.abs(m - ref_mean_x).sum()
    den = np.abs(ref_mean_x).sum() + 1e-12
    return float(num/den)

import matplotlib.pyplot as plt

def plot_positions(samples_dict_x, fname="sensors_post.png", n_show=3000):
    cols = len(samples_dict_x)
    fig, axes = plt.subplots(1, cols, figsize=(4*cols, 3), sharex=True, sharey=True)
    if cols == 1: axes = [axes]
    colors = plt.cm.tab10(np.arange(N_UNK))  # per-unknown color
    for ax, (name, Xs) in zip(axes, samples_dict_x.items()):
        S = Xs[:n_show]
        for k in range(N_UNK):
            pts = S[:, 2*k:2*k+2]
            ax.scatter(pts[:,0], pts[:,1], s=2, alpha=0.15, color=colors[k])
        ax.scatter(anchors[:,0], anchors[:,1], color='k', s=12, label="anchors")
        ax.scatter(theta_true[N_ANCH:,0], theta_true[N_ANCH:,1], marker='x', color='k', s=14, label="true unk")
        mean_x = S.mean(axis=0).reshape(N_UNK,2)
        ax.scatter(mean_x[:,0], mean_x[:,1], facecolors='none', edgecolors='lime', s=42, linewidths=1.5, label="post mean")
        ax.set_xlim(0, LIM); ax.set_ylim(0, LIM); ax.set_title(name, fontsize=10)
    axes[0].legend(loc='upper right', fontsize=7)
    plt.tight_layout(); plt.savefig(fname, dpi=180); plt.close()

def plot_bars(times, rems, out="sensors_bars.png"):
    names = list(times.keys())
    tvals = [times[k] for k in names]
    rvals = [rems[k] for k in names]
    fig, ax = plt.subplots(1,2, figsize=(10,3))
    ax[0].barh(names, tvals); ax[0].set_xlabel("time (s)")
    ax[1].barh(names, rvals); ax[1].set_xlabel("REM (↓)")
    plt.tight_layout(); plt.savefig(out, dpi=180); plt.close()

# ===================== Run all methods =====================
if __name__ == "__main__":
    results = {}
    samples_x = {}

    # Build the ground-truth reference in x-space (clip to (0,1) to match x = sigmoid(u))
    x_true_ref_vec = np.clip(theta_true[N_ANCH:].reshape(-1), 0.0, 1.0)

    # 1) HMC / NUTS
    print("\n=== HMC (NUTS) ===")
    t0 = time.time()
    U_hmc = run_hmc(draws=2500, tune=1000)
    t_hmc = time.time() - t0
    X_hmc = sigmoid_np(U_hmc)
    rem_hmc = REM_x(X_hmc, x_true_ref_vec)
    results["HMC"] = {"time": t_hmc, "REM": rem_hmc}
    samples_x["HMC"] = X_hmc
    print(f"HMC done in {t_hmc:.1f}s, samples: {U_hmc.shape}")

    # 2) MFVI-ADVI
    print("\n=== MFVI-ADVI ===")
    t0 = time.time()
    U_mfvi, t_mfvi_fit = run_advi(n_fit=20000, n_draw=4000)
    t_mfvi = time.time()-t0
    X_mfvi = sigmoid_np(U_mfvi)
    rem_mfvi = REM_x(X_mfvi, x_true_ref_vec)
    results["MFVI-ADVI"] = {"time": t_mfvi, "REM": rem_mfvi}
    samples_x["MFVI-ADVI"] = X_mfvi
    print(f"MFVI time {t_mfvi:.1f}s | REM {rem_mfvi:.3f}")

    # 3) GM-ADVI
    print("\n=== GM-ADVI (stabilized) ===")
    t0 = time.time()
    U_gm, t_gm_only = run_gmadvi(n_steps=3000, n_comp=24, T=8, lr=5e-3, n_draw=4000)
    t_gm = time.time()-t0 + t_gm_only
    X_gm = sigmoid_np(U_gm)
    rem_gm = REM_x(X_gm, x_true_ref_vec)
    results["GM-ADVI"] = {"time": t_gm, "REM": rem_gm}
    samples_x["GM-ADVI"] = X_gm
    print(f"GM-ADVI time {t_gm:.1f}s | REM {rem_gm:.3f}")

    # 4) S-ADVI
    print("\n=== S-ADVI ===")
    t0 = time.time()
    U_sa, t_sa_only = run_sadvi(n_steps=4000, lr=1e-2, K=4, n_draw=4000)
    t_sa = time.time()-t0 + t_sa_only
    X_sa = sigmoid_np(U_sa)
    rem_sa = REM_x(X_sa, x_true_ref_vec)
    results["S-ADVI"] = {"time": t_sa, "REM": rem_sa}
    samples_x["S-ADVI"] = X_sa
    print(f"S-ADVI time {t_sa:.1f}s | REM {rem_sa:.3f}")

    # 5) BVI (robust SPD + pure-JAX ELBO)
    print("\n=== BVI ===")
    t0 = time.time()
    U_bvi, t_bvi_only = run_bvi(T=12, iters_mode=250, lr=5e-2, n_draw=4000, hess_floor=1e-5)
    t_bvi = time.time()-t0 + t_bvi_only
    X_bvi = sigmoid_np(U_bvi)
    rem_bvi = REM_x(X_bvi, x_true_ref_vec)
    results["BVI"] = {"time": t_bvi, "REM": rem_bvi}
    samples_x["BVI"] = X_bvi
    print(f"BVI time {t_bvi:.1f}s | REM {rem_bvi:.3f}")

    # 6) LMA (Laplace Mixture Approx.)
    print("\n=== LMA (Laplace Mixture Approx.) ===")
    t0 = time.time()
    U_lma, t_lma_only = run_lma(n_starts=16, iters=600, lr=5e-2, kappa=1.0, n_draw=4000)
    t_lma = time.time() - t0 + t_lma_only
    X_lma = sigmoid_np(U_lma)
    rem_lma = REM_x(X_lma, x_true_ref_vec)
    results["LMA"] = {"time": t_lma, "REM": rem_lma}
    samples_x["LMA"] = X_lma
    print(f"LMA time {t_lma:.1f}s | REM {rem_lma:.3f}")

    # 7) EM-GMA
    print("\n=== EM-GMA ===")
    t0 = time.time()
    w_em, mu_em, Sig_em = em_gma_population(logf_u_numpy, K=24, M_bank=8192, n_iter=60, init_scale=0.9)
    t_em = time.time()-t0
    U_em = sample_gmm(4000, w_em, mu_em, Sig_em)
    X_em = sigmoid_np(U_em)
    rem_em = REM_x(X_em, x_true_ref_vec)
    results["EM-GMA"] = {"time": t_em, "REM": rem_em}
    samples_x["EM-GMA"] = X_em
    print(f"EM-GMA time {t_em:.1f}s | REM {rem_em:.3f}")

    # --------- Report ----------
    print("\n=== Time & REM (↓) vs GROUND TRUTH (x-space) ===")
    for k,v in results.items():
        print(f"{k:10s}  time {v['time']:6.1f}s  REM {v['REM']:.4f}")

    # Plots
    order = ["HMC","EM-GMA","BVI","LMA","GM-ADVI","S-ADVI","MFVI-ADVI"]
    plot_positions({k: samples_x[k] for k in order if k in samples_x}, "sensors_post.png")
    plot_bars({k:results[k]["time"] for k in order if k in results},
              {k:results[k]["REM"] for k in results}, "sensors_bars.png")
    print("Saved: sensors_post.png, sensors_bars.png")

"""# new."""

# -*- coding: utf-8 -*-
# sensor_localization_vi_all_independent_fast_emgma.py
# ------------------------------------------------------------
# Sensor Network Localization benchmark
# Methods (independent inits): HMC/NUTS, MFVI-ADVI, GM-ADVI, S-ADVI, BVI, LMA, EM-GMA (fast old version)
# Parameterization in u-space with x = LIM * sigmoid(u) ∈ (0, LIM)^D
# Uniform(x) prior => p(u) ∝ σ(u)[1-σ(u)] (LIM is a constant Jacobian factor)
# Links Z_ij ~ Bernoulli(exp(-d^2/(2R^2))) with optional exact |obs|=P_OBS
# ------------------------------------------------------------

import time, math, warnings
import numpy as np
warnings.filterwarnings("ignore", category=FutureWarning)

SEED = 123
rng_np = np.random.default_rng(SEED)

# ===================== Problem constants ====================
N = 11
N_ANCH = 5
N_UNK = N - N_ANCH         # 6 unknown sensors
DIMS = 2 * N_UNK           # 12 unknown coordinates

LIM   = 1.2                # region: [0, LIM]^2
R     = 0.3                # link-length scale in p(Z=1 | d)
SIGMA = 0.02               # range noise std
P_OBS = 14                 # target number of observed links

anchors = np.array([
    [0.08, 0.08],
    [1.12, 0.08],
    [0.08, 1.12],
    [1.12, 1.12],
    [0.60, 0.60],
], dtype=np.float32)

theta_true = np.zeros((N, 2), dtype=np.float32)
theta_true[:N_ANCH] = anchors
theta_true[N_ANCH:] = rng_np.uniform(0.05, LIM-0.05, size=(N_UNK, 2)).astype(np.float32)

pairs = [(i, j) for i in range(N) for j in range(i+1, N)]
P = len(pairs)

def pairwise_dist(Theta):
    D = np.zeros((N, N), dtype=np.float32)
    for (i, j) in pairs:
        d = float(np.linalg.norm(Theta[i] - Theta[j]))
        D[i, j] = D[j, i] = d
    return D

def gen_data(theta, exact_count=True, target=P_OBS, max_tries=5000):
    D_true = pairwise_dist(theta)
    Pi = np.exp(- (D_true**2) / (2.0 * R**2))
    attempt = 0
    while True:
        attempt += 1
        Z = np.zeros((N, N), dtype=int)
        for (i, j) in pairs:
            if rng_np.random() < Pi[i, j]:
                Z[i, j] = Z[j, i] = 1
        cnt = int(np.sum(Z) // 2)
        if not exact_count or cnt == target or attempt >= max_tries:
            break
    if exact_count and cnt != target:
        order = sorted(pairs, key=lambda ij: Pi[ij[0], ij[1]], reverse=True)
        Z[:] = 0
        for k in range(target):
            i, j = order[k]
            Z[i, j] = Z[j, i] = 1
    Y = np.zeros((N, N), dtype=np.float32)
    for (i, j) in pairs:
        if Z[i, j] == 1:
            yij = rng_np.normal(D_true[i, j], SIGMA)
            Y[i, j] = Y[j, i] = max(0.0, yij)
        else:
            Y[i, j] = Y[j, i] = 0.0
    obs_pairs = [(i, j) for (i, j) in pairs if Z[i, j] == 1]
    miss_pairs = [(i, j) for (i, j) in pairs if Z[i, j] == 0]
    return Z, Y, obs_pairs, miss_pairs

Z, Y, obs_pairs, miss_pairs = gen_data(theta_true, exact_count=True, target=P_OBS)
print(f"Data: N={N} (anchors={N_ANCH}, unknown={N_UNK}), |obs|={len(obs_pairs)}, |miss|={len(miss_pairs)}")

def unpack_x(x_vec):
    out = np.zeros((N, 2), dtype=np.float32)
    out[:N_ANCH] = anchors
    out[N_ANCH:] = x_vec.reshape(N_UNK, 2)
    return out

def sigmoid_np(u):
    return 1.0 / (1.0 + np.exp(-u))

# ================== Likelihood in x-space (NumPy) =====================
def loglik_x_numpy(Theta):
    ll = 0.0
    const_norm = math.log(SIGMA * math.sqrt(2.0 * math.pi))
    for (i, j) in obs_pairs:
        d = np.linalg.norm(Theta[i] - Theta[j]) + 1e-12
        p1 = math.exp(- (d*d) / (2.0 * R*R))
        ll += math.log(max(p1, 1e-300))
        ll += -0.5*((Y[i, j] - d)**2)/(SIGMA**2) - const_norm
    for (i, j) in miss_pairs:
        d = np.linalg.norm(Theta[i] - Theta[j]) + 1e-12
        p1 = math.exp(- (d*d) / (2.0 * R*R))
        ll += math.log(max(1.0 - p1, 1e-300))
    return ll

# ====================== PyMC (PyTensor) pieces ========================
import pymc as pm
import pytensor.tensor as at

def at_sigmoid(x):
    return at.where(at.ge(x, 0), 1.0 / (1.0 + at.exp(-x)), at.exp(x) / (1.0 + at.exp(x)))

def sensor_loglike_at_u(u_vec_at):
    x_unit = at_sigmoid(u_vec_at)
    x = LIM * x_unit
    x2 = x.reshape((N_UNK, 2))
    anchors_const = at.as_tensor_variable(anchors.astype(np.float32))
    Theta = at.concatenate([anchors_const, x2], axis=0)
    idx_i = np.array([i for (i, j) in pairs], dtype="int64")
    idx_j = np.array([j for (i, j) in pairs], dtype="int64")
    Theta_i = Theta[idx_i]; Theta_j = Theta[idx_j]
    diff = Theta_i - Theta_j
    d = at.sqrt(at.sum(diff**2, axis=1) + 1e-12)
    p1 = at.exp(-(d**2) / (2.0 * (R**2)))
    obs_mask = np.array([Z[i, j] == 1 for (i, j) in pairs], dtype=bool)
    mis_mask = ~obs_mask
    d_obs = d[obs_mask]
    p1_obs = p1[obs_mask]
    y_obs_np = np.array([Y[i, j] for (i, j) in pairs if Z[i, j] == 1], dtype=np.float32)
    y_obs = at.as_tensor_variable(y_obs_np)
    const_norm = float(np.log(SIGMA * np.sqrt(2.0 * np.pi)))
    ll_obs = at.sum(at.log(at.clip(p1_obs, 1e-300, 1e300))
                    - 0.5 * ((y_obs - d_obs) ** 2) / (SIGMA**2) - const_norm)
    p1_mis = p1[mis_mask]
    ll_mis = at.sum(at.log(at.clip(1.0 - p1_mis, 1e-300, 1e300)))
    return ll_obs + ll_mis

def log_uniform_prior_in_u_at(u):
    sx = at_sigmoid(u)
    return at.sum(at.log(sx + 1e-12) + at.log(1.0 - sx + 1e-12))

def run_hmc(draws=2500, tune=1000):
    with pm.Model() as model:
        u = pm.Normal("u", mu=0.0, sigma=10.0, shape=DIMS)
        base_logp = pm.logp(pm.Normal.dist(0.0, 10.0), u)
        pm.Potential("tilt", sensor_loglike_at_u(u) + log_uniform_prior_in_u_at(u) - base_logp)
        trace = pm.sample(
            draws=draws, tune=tune, chains=1, cores=1,
            step=pm.NUTS(target_accept=0.9), init="jitter+adapt_diag",
            random_seed=SEED, progressbar=True
        )
    U = trace.posterior["u"].to_numpy().squeeze()
    return U.reshape(-1, DIMS) if U.ndim == 3 else U

def run_advi(n_fit=20000, n_draw=4000):
    with pm.Model() as model:
        u = pm.Normal("u", mu=0.0, sigma=10.0, shape=DIMS)
        base_logp = pm.logp(pm.Normal.dist(0.0, 10.0), u)
        pm.Potential("tilt", sensor_loglike_at_u(u) + log_uniform_prior_in_u_at(u) - base_logp)
        t0 = time.time()
        approx = pm.fit(n=n_fit, method="advi", random_seed=SEED+1)
        spent = time.time() - t0
        tr = approx.sample(n_draw, random_seed=SEED+2)
    U = tr.posterior["u"].to_numpy().squeeze()
    U = U.reshape(-1, DIMS) if U.ndim == 3 else U
    return U, spent

# =============================== JAX setup ===============================
import jax
import jax.numpy as jnp
from jax.scipy.special import logsumexp as jls
import optax

key = jax.random.PRNGKey(SEED)
pairs_arr = jnp.array(pairs, dtype=jnp.int32)
obs_arr = jnp.array(obs_pairs, dtype=jnp.int32)
mis_arr = jnp.array(miss_pairs, dtype=jnp.int32)
anchors_j = jnp.array(anchors, dtype=jnp.float32)
Y_j = jnp.array(Y, dtype=jnp.float32)
R_J = jnp.float32(R)
SIGMA_J = jnp.float32(SIGMA)
LIM_J = jnp.float32(LIM)

def sigmoid_j(x):
    return jnp.where(x >= 0, 1/(1+jnp.exp(-x)), jnp.exp(x)/(1+jnp.exp(x)))

def unpack_u_to_full_x(u_vec):
    x = LIM_J * sigmoid_j(u_vec).reshape(N_UNK, 2)
    return jnp.concatenate([anchors_j, x], axis=0)

def loglik_u_j(u_vec):
    Theta = unpack_u_to_full_x(u_vec)
    def d_ij(idx):
        i,j = idx[0], idx[1]
        return jnp.linalg.norm(Theta[i]-Theta[j]) + 1e-12
    d_obs = jax.vmap(d_ij)(obs_arr)
    p1_obs = jnp.exp(- (d_obs**2) / (2.0 * R_J**2))
    y_obs = Y_j[tuple(obs_arr.T)]
    ll_obs = jnp.sum(jnp.log(jnp.clip(p1_obs, 1e-300, None))
                     - 0.5*((y_obs - d_obs)**2)/(SIGMA_J**2)
                     - jnp.log(SIGMA_J*jnp.sqrt(2*jnp.pi)))
    d_mis = jax.vmap(d_ij)(mis_arr)
    p1_mis = jnp.exp(- (d_mis**2) / (2.0 * R_J**2))
    ll_mis = jnp.sum(jnp.log(jnp.clip(1.0 - p1_mis, 1e-300, None)))
    return ll_obs + ll_mis

def logprior_u_j(u_vec):
    s = sigmoid_j(u_vec)
    return jnp.sum(jnp.log(s + 1e-12) + jnp.log(1.0 - s + 1e-12))

def logf_u_j(u_vec):
    val = loglik_u_j(u_vec) + logprior_u_j(u_vec)
    return jnp.nan_to_num(val, neginf=-1e30, posinf=1e30)

grad_logf_u = jax.grad(logf_u_j)

# ===================== GM-ADVI (stabilized, u-space) =====================
def run_gmadvi(n_steps=3000, n_comp=24, T=4, lr=1e-3, n_draw=4000):
    key = jax.random.PRNGKey(SEED+10)
    d = DIMS
    means0 = 0.0 + 0.15*jax.random.normal(key, (n_comp, d))
    log_scales0 = jnp.log(0.2 + 0.1*jax.random.uniform(key, (n_comp, d)))
    logits0 = jnp.zeros((n_comp,))
    params = {"logits": logits0, "means": means0, "log_scales": log_scales0}
    def scales(p): return 1e-2 + jax.nn.softplus(p["log_scales"])
    def logq(z, p):
        s = scales(p)
        logw = jax.nn.log_softmax(p["logits"])
        comp = -0.5*jnp.sum(((z - p["means"])/s)**2 + 2*jnp.log(s) + jnp.log(2*jnp.pi), axis=-1)
        return jls(logw + comp, axis=-1)
    def siwae_loss(p, key):
        key, sub = jax.random.split(key)
        eps = jax.random.normal(sub, (T, n_comp, d))
        z = p["means"][None,:,:] + scales(p)[None,:,:]*eps
        zf = z.reshape(-1, d)
        lp = jax.vmap(logf_u_j)(zf).reshape(T, n_comp)
        lq = jax.vmap(lambda row: logq(row, p))(zf).reshape(T, n_comp)
        lw = jax.nn.log_softmax(p["logits"])[None,:] + lp - lq
        obj = jax.scipy.special.logsumexp(lw, axis=1) - jnp.log(T)
        ent_reg = 1e-3*jnp.sum(jax.nn.softmax(p["logits"]) * (jnp.log(jax.nn.softmax(p["logits"])+1e-12)))
        scale_reg = 1e-4*jnp.mean(scales(p)**2)
        return -jnp.mean(obj) + ent_reg + scale_reg
    opt = optax.chain(optax.clip_by_global_norm(5.0), optax.adam(lr))
    state = opt.init(params)
    @jax.jit
    def step(p, st, key):
        loss, g = jax.value_and_grad(siwae_loss)(p, key)
        upd, st = opt.update(g, st)
        p = optax.apply_updates(p, upd)
        return p, st, loss
    t0 = time.time()
    for _ in range(n_steps):
        key, sub = jax.random.split(key)
        params, state, _ = step(params, state, sub)
    spent = time.time()-t0
    key, ckey, nkey = jax.random.split(key, 3)
    alphas = jax.nn.softmax(params["logits"])
    comps = jax.random.choice(ckey, n_comp, (n_draw,), p=alphas)
    z = jax.random.normal(nkey, (n_draw, d))* (1e-2 + jax.nn.softplus(params["log_scales"]))[comps] + params["means"][comps]
    return np.array(z), spent

# ===================== SPD utilities =====================
def nearest_spd(A: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    A = 0.5 * (A + A.T)
    w, V = np.linalg.eigh(A)
    w = np.maximum(w, floor)
    return (V * w) @ V.T

def safe_cholesky(S: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    try:
        return np.linalg.cholesky(S)
    except np.linalg.LinAlgError:
        S2 = nearest_spd(S, floor=floor)
        return np.linalg.cholesky(S2)

def cap_condition_number(S: np.ndarray, kmax: float = 1e6, floor: float = 1e-8) -> np.ndarray:
    S = 0.5*(S+S.T)
    w, V = np.linalg.eigh(S)
    w = np.clip(w, floor, None)
    wmax = np.max(w)
    w = np.maximum(w, wmax / kmax)
    return (V * w) @ V.T

# ===================== BVI (independent, recent) =====================
class BVIMix:
    def __init__(self, d, init_scale=0.8):
        self.w = [1.0]
        self.means = [np.zeros((d,), dtype=float)]
        self.covs = [np.eye(d, dtype=float) * (init_scale**2)]
        self.d = d
    def logq(self, z):
        logws = jnp.log(jnp.array(self.w))
        comp = []
        for m, S in zip(self.means, self.covs):
            S_j = jnp.array(S)
            diff = z - jnp.array(m)
            sign, logdet = jnp.linalg.slogdet(S_j)
            sol = jnp.linalg.solve(S_j, diff)
            comp.append(-0.5*jnp.sum(diff*sol, axis=-1) - 0.5*self.d*jnp.log(2*jnp.pi) - 0.5*logdet)
        comp = jnp.stack(comp, axis=-1)
        return jls(logws + comp, axis=-1)
    def add_component(self, mean: np.ndarray, cov: np.ndarray, alpha: float, floor: float = 1e-6, kcap: float = 1e6):
        alpha = float(alpha)
        self.w = [(1-alpha)*w for w in self.w] + [alpha]
        cov = cap_condition_number(nearest_spd(np.array(cov, dtype=float), floor=floor), kmax=kcap, floor=floor)
        self.means.append(np.array(mean, dtype=float))
        self.covs.append(cov)

def run_bvi(T=12, iters_mode=250, lr=5e-2, alpha_grid=None, n_draw=4000, hess_floor=1e-5):
    if alpha_grid is None:
        alpha_grid = np.linspace(0.05, 0.5, 10)
    d = DIMS
    mix = BVIMix(d, init_scale=0.8)
    t0 = time.time()
    for t in range(T):
        v = 0.0 + 0.2*jax.random.normal(jax.random.PRNGKey(SEED+100+t), (d,))
        opt = optax.adam(lr)
        st = opt.init(v)
        @jax.jit
        def step(v, st):
            loss, g = jax.value_and_grad(lambda x: -(logf_u_j(x) - mix.logq(x)))(v)
            upd, st = opt.update(g, st)
            v = optax.apply_updates(v, upd)
            return v, st, loss
        for _ in range(iters_mode):
            v, st, _ = step(v, st)
        H = jax.hessian(lambda z: -(logf_u_j(z) - mix.logq(z)))(v)
        H = np.array(H, dtype=float)
        Prec = nearest_spd(H, floor=hess_floor)
        cov = np.linalg.inv(Prec)
        cov = cap_condition_number(nearest_spd(cov, floor=hess_floor), kmax=1e6, floor=hess_floor)
        cov_j = jnp.array(cov); v_j = jnp.array(v)
        def logq_new(z, a):
            sign, logdet = jnp.linalg.slogdet(cov_j)
            comp_new = -0.5*jnp.sum((z - v_j) * jnp.linalg.solve(cov_j, (z - v_j)), axis=-1) \
                       - 0.5*d*jnp.log(2*jnp.pi) - 0.5*logdet
            logmixw = jnp.stack([jnp.log(1.0 - a + 1e-12), jnp.log(a + 1e-12)])
            two_terms = jnp.stack([mix.logq(z), comp_new], axis=-1)
            return jls(logmixw + two_terms, axis=-1)
        zs = jax.random.normal(jax.random.PRNGKey(SEED+200+t), (1024, d)) @ jnp.linalg.cholesky(cov_j).T + v_j
        scores = []
        for a in alpha_grid:
            s = jnp.mean(jax.vmap(logf_u_j)(zs) - jax.vmap(lambda zi: logq_new(zi, jnp.float32(a)))(zs))
            scores.append(float(s))
        a_best = float(alpha_grid[int(np.argmax(np.array(scores)))])
        mix.add_component(np.array(v), cov, a_best, floor=hess_floor, kcap=1e6)
    spent = time.time()-t0
    w = np.array(mix.w)/np.sum(mix.w)
    comps = rng_np.choice(len(w), size=n_draw, p=w)
    z = np.zeros((n_draw, d))
    for k in range(len(w)):
        idx = np.where(comps==k)[0]
        if idx.size:
            Lc = safe_cholesky(mix.covs[k], floor=hess_floor)
            eps = rng_np.normal(size=(idx.size, d))
            z[idx] = mix.means[k] + eps @ Lc.T
    return z, spent

# ===================== LMA (independent, recent) =====================
def lma_find_modes(n_starts=16, iters=600, lr=5e-2, tol_merge=1e-2):
    modes, vals = [], []
    opt = optax.chain(optax.clip_by_global_norm(20.0), optax.adam(lr))
    for s in range(n_starts):
        key = jax.random.PRNGKey(SEED + 300 + s)
        u = 0.0 + 0.5 * jax.random.normal(key, (DIMS,))
        state = opt.init(u)
        @jax.jit
        def step(u, st):
            loss, g = jax.value_and_grad(lambda v: -logf_u_j(v))(u)
            updates, st = opt.update(g, st)
            u = optax.apply_updates(u, updates)
            return u, st, loss
        for _ in range(iters):
            u, state, _ = step(u, state)
        u_np = np.array(u)
        val = float(logf_u_j(u_np))
        keep = True
        x_u = LIM * sigmoid_np(u_np)
        for m in modes:
            if np.linalg.norm(LIM * sigmoid_np(m) - x_u) < tol_merge:
                keep = False; break
        if keep:
            modes.append(u_np); vals.append(val)
    idx = np.argsort(-np.array(vals))
    modes = [modes[i] for i in idx]
    vals  = [vals[i]  for i in idx]
    return modes, vals

def lma_build_mixture(modes, vals, kappa=1.0, floor=1e-6):
    comps, log_evid = [], []
    d = DIMS
    for u_mode, val in zip(modes, vals):
        H = np.array(jax.hessian(logf_u_j)(u_mode), dtype=float)
        Prec = nearest_spd(-H, floor=floor)
        Sigma = np.linalg.inv(Prec)
        Sigma = cap_condition_number(nearest_spd((kappa**2) * Sigma, floor=floor), kmax=1e6, floor=floor)
        sign, logdet = np.linalg.slogdet(Sigma)
        log_evid.append(val + 0.5*d*math.log(2*math.pi) + 0.5*logdet)
        comps.append({"mu": np.array(u_mode, dtype=float), "Sigma": Sigma})
    log_evid = np.array(log_evid); m = log_evid.max()
    w = np.exp(log_evid - m); w /= w.sum()
    for c, wi in zip(comps, w): c["w"] = float(wi)
    return comps

def lma_sample(comps, n_draw=4000, rng=rng_np, floor=1e-8):
    w = np.array([c["w"] for c in comps], dtype=float)
    J = rng.choice(len(comps), size=n_draw, p=w)
    U = np.zeros((n_draw, DIMS), dtype=float)
    for k in range(len(comps)):
        idx = np.where(J==k)[0]
        if idx.size:
            L = safe_cholesky(comps[k]["Sigma"], floor=floor)
            eps = rng.normal(size=(idx.size, DIMS))
            U[idx] = comps[k]["mu"] + eps @ L.T
    return U

def run_lma(n_starts=16, iters=600, lr=5e-2, kappa=1.0, n_draw=4000):
    t0 = time.time()
    modes, vals = lma_find_modes(n_starts=n_starts, iters=iters, lr=lr)
    comps = lma_build_mixture(modes, vals, kappa=kappa)
    U = lma_sample(comps, n_draw=n_draw, rng=rng_np)
    spent = time.time() - t0
    return U, spent

# ===================== EM-GMA (FAST, OLD-STYLE population EM) =====================
def make_spd(S, eps=1e-6, kcap=1e6):
    S = nearest_spd(S, floor=eps)
    return cap_condition_number(S, kmax=kcap, floor=eps)

def gmm_logpdf(X, w, mus, Sigmas, chols=None):
    K = len(w); d = X.shape[1]
    if chols is None:
        chols = [np.linalg.cholesky(Sigmas[k]) for k in range(K)]
    parts = np.stack([
        np.log(max(w[k], 1e-300)) - 0.5*np.sum((np.linalg.solve(chols[k], (X - mus[k]).T))**2, axis=0)
        - 0.5*d*np.log(2*np.pi) - np.log(np.diag(chols[k])).sum()
        for k in range(K)
    ], axis=1)
    m = np.max(parts, axis=1, keepdims=True)
    log_qz = (m + np.log(np.sum(np.exp(parts - m), axis=1, keepdims=True))).squeeze(1)
    return log_qz, parts, chols

def sample_gmm(n, w, mus, Sigmas, rng=rng_np):
    K, d = mus.shape
    w = np.array(w, dtype=float); w = w / np.sum(w)
    comps = rng.choice(K, size=n, p=w)
    X = np.empty((n, d))
    for k in range(K):
        idx = np.where(comps==k)[0]
        if idx.size:
            L = np.linalg.cholesky(Sigmas[k])
            eps = rng.normal(size=(idx.size, d))
            X[idx] = mus[k] + eps @ L.T
    return X

def logf_u_numpy(u_vec):
    x_unit = sigmoid_np(u_vec)
    x = LIM * x_unit
    Theta = unpack_x(x)
    return loglik_x_numpy(Theta) + np.sum(np.log(x_unit + 1e-12) + np.log(1.0 - x_unit + 1e-12))

def em_gma_population_fast(log_f, K=24, M_bank=4096, n_iter=40, ridge=1e-5, init_scale=0.9, seed=SEED+40):
    """
    OLD / FAST population EM for inclusive KL:
      - independent random init (uniform weights; means ~ N(0, init_scale^2 I); Σ = I)
      - fixed-size proposal bank (no ESS gating)
      - no annealing; full-covariance updates with SPD guard
    """
    rng_local = np.random.default_rng(seed)
    d = DIMS
    w = np.ones(K)/K
    mus = 0.0 + init_scale*rng_local.normal(size=(K, d))
    Sigmas = np.stack([np.eye(d) for _ in range(K)], axis=0)

    for _ in range(n_iter):
        # draw proposal bank from current q
        Zb = sample_gmm(M_bank, w, mus, Sigmas, rng_local)
        log_qz, parts, _ = gmm_logpdf(Zb, w, mus, Sigmas)
        log_p  = np.array([log_f(z) for z in Zb])
        # SNIS weights
        lw = log_p - log_qz
        lw -= lw.max()
        omega = np.exp(lw); omega /= (omega.sum() + 1e-16)
        # responsibilities under q
        Rm = np.exp(parts - log_qz[:,None])  # (M_bank x K)

        # population EM updates (SNIS-approximated)
        Nk = (omega[:,None] * Rm).sum(axis=0) + 1e-16
        w  = Nk / Nk.sum()
        mus = ((omega[:,None] * Rm).T @ Zb) / Nk[:,None]
        for k in range(K):
            Zc = Zb - mus[k]
            Sk = (omega[:,None] * Rm[:,[k]] * Zc).T @ Zc / Nk[k]
            Sigmas[k] = make_spd(Sk + ridge*np.eye(d))

    return w, mus, Sigmas

# ===================== Utilities: REM, plotting =====================
def REM_x(est_samps_x, ref_mean_x):
    m = est_samps_x.mean(axis=0)
    num = np.abs(m - ref_mean_x).sum()
    den = np.abs(ref_mean_x).sum() + 1e-12
    return float(num/den)

import matplotlib.pyplot as plt

# def plot_positions(samples_dict_x, fname="sensors_post.png", n_show=3000):
#     cols = len(samples_dict_x)
#     fig, axes = plt.subplots(1, cols, figsize=(4*cols, 3), sharex=True, sharey=True)
#     if cols == 1: axes = [axes]
#     colors = plt.cm.tab10(np.arange(N_UNK))
#     for ax, (name, Xs) in zip(axes, samples_dict_x.items()):
#         S = Xs[:n_show]
#         for k in range(N_UNK):
#             pts = S[:, 2*k:2*k+2]
#             ax.scatter(pts[:,0], pts[:,1], s=2, alpha=0.15, color=colors[k])
#         ax.scatter(anchors[:,0], anchors[:,1], color='k', s=12, label="anchors")
#         ax.scatter(theta_true[N_ANCH:,0], theta_true[N_ANCH:,1], marker='x', color='k', s=14, label="true unk")
#         mean_x = S.mean(axis=0).reshape(N_UNK,2)
#         ax.scatter(mean_x[:,0], mean_x[:,1], facecolors='none', edgecolors='lime', s=42, linewidths=1.5, label="post mean")
#         ax.set_xlim(0, LIM); ax.set_ylim(0, LIM); ax.set_title(name, fontsize=10)
#     axes[0].legend(loc='upper right', fontsize=7)
#     plt.tight_layout(); plt.savefig(fname, dpi=180); plt.close()

def plot_positions(samples_dict_x, fname="sensors_post.png", n_show=3000):
    import numpy as np
    names_items = list(samples_dict_x.items())
    n = len(names_items)
    nrows = 2 if n > 1 else 1
    ncols = int(np.ceil(n / nrows))

    fig, axes = plt.subplots(nrows, ncols, figsize=(4*ncols, 3*nrows), sharex=True, sharey=True)
    axes = np.array(axes).reshape(-1)  # flatten to 1D list of axes

    colors = plt.cm.tab10(np.arange(N_UNK))
    for idx, (name, Xs) in enumerate(names_items):
        ax = axes[idx]
        S = Xs[:n_show]
        for k in range(N_UNK):
            pts = S[:, 2*k:2*k+2]
            ax.scatter(pts[:, 0], pts[:, 1], s=4, alpha=0.15, color=colors[k])
        ax.scatter(anchors[:, 0], anchors[:, 1], color='k', s=14, label="anchors")
        ax.scatter(theta_true[N_ANCH:, 0], theta_true[N_ANCH:, 1], marker='x', color='k', s=16, label="true unk")
        mean_x = S.mean(axis=0).reshape(N_UNK, 2)
        ax.scatter(mean_x[:, 0], mean_x[:, 1], facecolors='none', edgecolors='lime', s=42, linewidths=1.5, label="post mean")
        ax.set_xlim(0, LIM); ax.set_ylim(0, LIM); ax.set_title(name, fontsize=10)

    # hide any unused axes (e.g., when n is odd)
    for j in range(len(names_items), len(axes)):
        axes[j].axis('off')

    axes[0].legend(loc='upper right', fontsize=7)
    plt.tight_layout()
    plt.savefig(fname, dpi=180)
    plt.close()

def plot_bars(times, rems, out="sensors_bars.png"):
    names = list(times.keys())
    tvals = [times[k] for k in names]
    rvals = [rems[k] for k in names]
    fig, ax = plt.subplots(1,2, figsize=(10,3))
    ax[0].barh(names, tvals); ax[0].set_xlabel("time (s)")
    ax[1].barh(names, rvals); ax[1].set_xlabel("REM (↓)")
    plt.tight_layout(); plt.savefig(out, dpi=180); plt.close()

# ===================== Run all methods (independent; fast EM-GMA) =====================
if __name__ == "__main__":
    results = {}
    samples_x = {}
    x_true_ref_vec = theta_true[N_ANCH:].reshape(-1)

    print("\n=== HMC (NUTS) ===")
    t0 = time.time(); U_hmc = run_hmc(draws=4000, tune=1000); t_hmc = time.time() - t0
    X_hmc = LIM * sigmoid_np(U_hmc)
    rem_hmc = REM_x(X_hmc, x_true_ref_vec)
    results["HMC"] = {"time": t_hmc, "REM": rem_hmc}; samples_x["HMC"] = X_hmc
    print(f"HMC time {t_hmc:.1f}s | REM {rem_hmc:.3f}")

    print("\n=== MFVI-ADVI ===")
    t0 = time.time(); U_mfvi, _ = run_advi(n_fit=20000, n_draw=4000); t_mfvi = time.time()-t0
    X_mfvi = LIM * sigmoid_np(U_mfvi)
    rem_mfvi = REM_x(X_mfvi, x_true_ref_vec)
    results["MFVI-ADVI"] = {"time": t_mfvi, "REM": rem_mfvi}; samples_x["MFVI-ADVI"] = X_mfvi
    print(f"MFVI time {t_mfvi:.1f}s | REM {rem_mfvi:.3f}")

    print("\n=== GM-ADVI (stabilized) ===")
    t0 = time.time(); U_gm, t_gm_only = run_gmadvi(n_steps=3000, n_comp=24, T=4, lr=1e-3, n_draw=4000)
    t_gm = time.time()-t0 + t_gm_only
    X_gm = LIM * sigmoid_np(U_gm)
    rem_gm = REM_x(X_gm, x_true_ref_vec)
    results["GM-ADVI"] = {"time": t_gm, "REM": rem_gm}; samples_x["GM-ADVI"] = X_gm
    print(f"GM-ADVI time {t_gm:.1f}s | REM {rem_gm:.3f}")

    print("\n=== S-ADVI ===")
    t0 = time.time()
    def run_sadvi(n_steps=4000, lr=1e-2, K=4, n_draw=4000):
        key = jax.random.PRNGKey(SEED+20)
        d = DIMS
        params = {
            "mu": jnp.zeros((d,)),
            "log_sigma": jnp.log(jnp.full((d,), 0.7)),
            "a": jnp.log(jnp.ones((d,K))*0.1),
            "b": jnp.log(jnp.ones((d,K))*0.5),
            "c": jnp.zeros((d,K))
        }
        def transform(eps, p):
            sig = 1e-3 + jax.nn.softplus(p["log_sigma"])
            A = jax.nn.softplus(p["a"]); B = jax.nn.softplus(p["b"]); C = p["c"]
            S = jax.nn.sigmoid(B*eps[...,None] + C)
            u = p["mu"] + sig*( eps + jnp.sum(A*S, axis=-1) )
            du_de = sig*( 1.0 + jnp.sum(A*B*S*(1.0-S), axis=-1) )
            return u, du_de
        def elbo(p, key, T=256):
            key, sub = jax.random.split(key)
            eps = jax.random.normal(sub, (T, d))
            u, jac = transform(eps, p)
            logq = -0.5*jnp.sum(eps**2, axis=-1) - 0.5*d*jnp.log(2*jnp.pi) - jnp.sum(jnp.log(jac), axis=-1)
            lp = jax.vmap(logf_u_j)(u)
            return jnp.mean(lp - logq)
        opt = optax.chain(optax.clip_by_global_norm(10.0), optax.adam(lr))
        state = opt.init(params)
        @jax.jit
        def step(p, st, key):
            loss, g = jax.value_and_grad(lambda pr: -elbo(pr, key))(p)
            upd, st = opt.update(g, st)
            p = optax.apply_updates(p, upd)
            return p, st, loss
        for _ in range(n_steps):
            key, sub = jax.random.split(key)
            params, state, _ = step(params, state, sub)
        key, sub = jax.random.split(key)
        eps = jax.random.normal(sub, (n_draw, d))
        u, _ = transform(eps, params)
        return np.array(u), 0.0
    U_sa, _ = run_sadvi(n_steps=4000, lr=1e-2, K=4, n_draw=4000)
    t_sa = time.time()-t0
    X_sa = LIM * sigmoid_np(U_sa)
    rem_sa = REM_x(X_sa, x_true_ref_vec)
    results["S-ADVI"] = {"time": t_sa, "REM": rem_sa}; samples_x["S-ADVI"] = X_sa
    print(f"S-ADVI time {t_sa:.1f}s | REM {rem_sa:.3f}")

    print("\n=== BVI (independent) ===")
    t0 = time.time(); U_bvi, t_bvi_only = run_bvi(T=12, iters_mode=250, lr=5e-2, n_draw=4000, hess_floor=1e-5)
    t_bvi = time.time()-t0 + t_bvi_only
    X_bvi = LIM * sigmoid_np(U_bvi)
    rem_bvi = REM_x(X_bvi, x_true_ref_vec)
    results["BVI"] = {"time": t_bvi, "REM": rem_bvi}; samples_x["BVI"] = X_bvi
    print(f"BVI time {t_bvi:.1f}s | REM {rem_bvi:.3f}")

    print("\n=== LMA (independent) ===")
    t0 = time.time(); U_lma, t_lma_only = run_lma(n_starts=16, iters=600, lr=5e-2, kappa=1.0, n_draw=4000)
    t_lma = time.time() - t0 + t_lma_only
    X_lma = LIM * sigmoid_np(U_lma)
    rem_lma = REM_x(X_lma, x_true_ref_vec)
    results["LMA"] = {"time": t_lma, "REM": rem_lma}; samples_x["LMA"] = X_lma
    print(f"LMA time {t_lma:.1f}s | REM {rem_lma:.3f}")

    print("\n=== EM-GMA (fast old version; independent init) ===")
    t0 = time.time()
    w_em, mu_em, Sig_em = em_gma_population_fast(
        logf_u_numpy, K=24, M_bank=4096, n_iter=40, ridge=1e-5, init_scale=0.9, seed=SEED+40
    )
    t_em = time.time()-t0
    U_em = sample_gmm(4000, w_em, mu_em, Sig_em)
    X_em = LIM * sigmoid_np(U_em)
    rem_em = REM_x(X_em, x_true_ref_vec)
    results["EM-GMA"] = {"time": t_em, "REM": rem_em}; samples_x["EM-GMA"] = X_em
    print(f"EM-GMA time {t_em:.1f}s | REM {rem_em:.3f}")

    print("\n=== Time & REM (↓) vs GROUND TRUTH (x-space) ===")
    for k,v in results.items():
        print(f"{k:10s}  time {v['time']:6.1f}s  REM {v['REM']:.4f}")

    order = ["HMC","EM-GMA","BVI","LMA","GM-ADVI","S-ADVI","MFVI-ADVI"]
    plot_positions({k: samples_x[k] for k in order if k in samples_x}, "sensors_post.png")
    plot_bars({k:results[k]['time'] for k in order if k in results},
              {k:results[k]['REM'] for k in results}, "sensors_bars.png")
    print("Saved: sensors_post.png, sensors_bars.png")

"""# End."""
