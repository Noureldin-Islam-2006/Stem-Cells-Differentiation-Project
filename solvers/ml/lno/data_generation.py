"""Data generation routines for the stiff genetic switch ODE system.
"""

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from scipy.integrate import solve_ivp
from scipy.optimize import fsolve

PARAMS = dict(
    a1  = 1.0,    # GATA-1 self-activation strength
    a2  = 1.0,    # PU.1   self-activation strength
    b1  = 0.5,    # GATA-1 mutual-inhibition amplitude
    b2  = 0.5,    # PU.1   mutual-inhibition amplitude
    th1 = 0.5,    # Hill threshold (self-activation)
    th2 = 0.5,    # Hill threshold (mutual-inhibition)
    n   = 4,      # Hill cooperativity (self-activation)
    m   = 4,      # Hill cooperativity (mutual-inhibition)
    k1  = 100.0,  # GATA-1 degradation — FAST -> λ_max ≈ -100
    k2  = 1.0,    # PU.1   degradation — slow -> λ_min ≈ -1
)

def _rhs(x, y, p):
    a1, a2 = p["a1"], p["a2"]
    b1, b2 = p["b1"], p["b2"]
    n,  m  = p["n"],  p["m"]
    k1, k2 = p["k1"], p["k2"]
    th1n   = p["th1"] ** n
    th2m   = p["th2"] ** m
    xn = abs(x) ** n;  yn = abs(y) ** n
    xm = abs(x) ** m;  ym = abs(y) ** m

    f_self_x = a1 * xn / (th1n + xn)
    f_self_y = a2 * yn / (th1n + yn)
    f_inh = b1 * th2m / (th2m + xm * ym)

    dxdt = f_self_x + f_inh - k1 * x
    dydt = f_self_y + f_inh - k2 * y
    return dxdt, dydt


def genetic_switch(t, state, p=PARAMS):
    """RHS with concentration clipping to prevent negative values."""
    x, y = max(state[0], 0.0), max(state[1], 0.0)
    return list(_rhs(x, y, p))


def genetic_switch_raw(t, state, p=PARAMS):
    """RHS without clipping (used to show FE blowup)."""
    x, y = state[0], state[1]
    return list(_rhs(x, y, p))


def forward_euler(f, y0, t_span, dt, blow_threshold=1e5):
    """Explicit Forward Euler solver."""
    t0, tf  = t_span
    t_list  = [t0]
    y_list  = [np.array(y0, dtype=float)]
    t       = t0
    y       = np.array(y0, dtype=float)
    blew_up = False
    blow_t  = tf

    while t < tf - 1e-12:
        dydt = np.array(f(t, y))
        y    = y + dt * dydt
        t   += dt
        t_list.append(t)
        y_list.append(y.copy())

        if np.any(np.abs(y) > blow_threshold) or np.any(np.isnan(y)):
            blew_up = True
            blow_t  = t
            break

    return np.array(t_list), np.array(y_list), blew_up, blow_t


def backward_euler(f, y0, t_span, dt):
    """Implicit Backward Euler solver."""
    t0, tf = t_span
    t_list = [t0]
    y_list = [np.array(y0, dtype=float)]
    t = t0
    y = np.array(y0, dtype=float)

    while t < tf - 1e-12:
        t_next = t + dt
        def residual(y_next):
            return y_next - y - dt * np.array(f(t_next, y_next))
        y_guess = y + dt * np.array(f(t, y))
        y_next  = fsolve(residual, y_guess)
        t = t_next
        y = y_next.copy()
        t_list.append(t)
        y_list.append(y.copy())

    return np.array(t_list), np.array(y_list)


def solve_bdf(y0, t_span=(0.0, 10.0), t_eval=None, rtol=1e-8, atol=1e-10):
    """Implicit variable-step BDF solver from scipy."""
    return solve_ivp(
        fun          = genetic_switch,
        t_span       = t_span,
        y0           = y0,
        method       = "BDF",
        t_eval       = t_eval,
        rtol         = rtol,
        atol         = atol,
        dense_output = False,
    )


def generate_dataset(n_traj=1000, t_end=10, n_steps=501, ic_range=(0.0, 3.0), seed=42, ic_path="ic_samples.npy", traj_path="trajectories.npy"):
    """Generate multiple trajectories using BDF for ML training."""
    rng    = np.random.default_rng(seed)
    t_eval = np.linspace(0, t_end, n_steps)

    ics    = rng.uniform(ic_range[0], ic_range[1], size=(n_traj, 2))
    trajs  = np.zeros((n_traj, n_steps, 2), dtype=np.float32)

    n_failed = 0
    for i, y0 in enumerate(ics):
        sol = solve_ivp(
            fun    = genetic_switch,
            t_span = (0.0, t_end),
            y0     = y0,
            method = "BDF",
            t_eval = t_eval,
            rtol   = 1e-6,
            atol   = 1e-8,
        )
        if sol.success:
            trajs[i] = sol.y.T.astype(np.float32)
        else:
            n_failed += 1
            trajs[i]  = np.nan

    np.save(ic_path,   ics)
    np.save(traj_path, trajs)
    return ics, trajs, t_eval


def plot_dataset_samples(ics, trajectories, t_eval, n_show=12, out_path="dataset_samples.png"):
    """Plot sample trajectories to check data generation."""
    rng = np.random.default_rng(7)
    idx = rng.choice(len(ics), size=min(n_show, len(ics)), replace=False)

    n_cols = 4
    n_rows = (len(idx) + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(16, 4 * n_rows))
    fig.patch.set_facecolor("#0d0d1a")

    for ax, i in zip(axes.flat, idx):
        ax.set_facecolor("#12122a")
        ax.plot(t_eval, trajectories[i, :, 0], color="#00e5ff", lw=1.4, label="x")
        ax.plot(t_eval, trajectories[i, :, 1], color="#ff4081", lw=1.4, label="y")
        ax.set_title(f"IC=({ics[i,0]:.2f},{ics[i,1]:.2f})", color="white", fontsize=8, pad=4)
        ax.tick_params(colors="#aaaacc", labelsize=6)
        ax.set_xlabel("t", color="#aaaacc", fontsize=7)
        ax.set_ylabel("conc.", color="#aaaacc", fontsize=7)
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a50")
        ax.grid(True, color="#1e1e3a", lw=0.5, ls="--")
        if ax is axes.flat[0]:
            ax.legend(fontsize=7, facecolor="#12122a", labelcolor="white", framealpha=0.9)

    for ax in axes.flat[len(idx):]:
        ax.set_visible(False)

    plt.tight_layout()
    plt.savefig(out_path, dpi=140, bbox_inches="tight", facecolor="#0d0d1a")
    plt.close()
