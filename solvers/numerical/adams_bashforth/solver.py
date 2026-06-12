"""Adams-Bashforth 2-step (AB2) ODE solver.

Uses a single RK4 step to bootstrap, then the explicit 2-step formula:
    y_{n+1} = y_n + (h/2) * (3·f(t_n, y_n) − f(t_{n-1}, y_{n-1}))
"""

import numpy as np

NAME = "Adams-Bashforth"
KEY = "adams_bashforth"
DESCRIPTION = "Explicit 2-step multi-step method. Uses previous function evaluations to extrapolate, bootstrapped with a single RK4 step."
IS_IMPLEMENTED = True


def _rk4_step(ode_func, t, y, h, args):
    """Single RK4 step used to bootstrap the multi-step method."""
    k1 = np.array(ode_func(t, y, *args))
    k2 = np.array(ode_func(t + 0.5 * h, y + 0.5 * h * k1, *args))
    k3 = np.array(ode_func(t + 0.5 * h, y + 0.5 * h * k2, *args))
    k4 = np.array(ode_func(t + h, y + h * k3, *args))
    return y + (h / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)


def solve(ode_func, t_span, y0, args, dt):
    """Solve the ODE system using Adams-Bashforth 2-step.

    Parameters
    ----------
    ode_func : callable  —  f(t, y, *args) → dy/dt
    t_span   : tuple     —  (t_start, t_end)
    y0       : array-like — initial conditions
    args     : tuple     —  extra arguments for *ode_func*
    dt       : float     —  step size

    Returns
    -------
    t : ndarray, shape (n_steps,)
    y : ndarray, shape (n_vars, n_steps)
    """
    t_start, t_end = t_span
    n_steps = int(np.ceil((t_end - t_start) / dt))
    t = np.linspace(t_start, t_end, n_steps + 1)
    n_vars = len(y0)
    y = np.zeros((n_vars, len(t)))
    y[:, 0] = y0

    # Evaluate slope at t0
    f_prev = np.array(ode_func(t[0], y[:, 0], *args))

    # Bootstrap: use a single RK4 step for y_1
    h0 = t[1] - t[0]
    y[:, 1] = _rk4_step(ode_func, t[0], y[:, 0], h0, args)
    f_curr = np.array(ode_func(t[1], y[:, 1], *args))

    # AB2 multi-step loop
    for i in range(1, n_steps):
        h = t[i + 1] - t[i]
        y[:, i + 1] = y[:, i] + (h / 2.0) * (3.0 * f_curr - f_prev)
        y[:, i + 1] = np.maximum(y[:, i + 1], 0.0)  # enforce non-negativity

        f_prev = f_curr
        f_curr = np.array(ode_func(t[i + 1], y[:, i + 1], *args))

    return t, y
