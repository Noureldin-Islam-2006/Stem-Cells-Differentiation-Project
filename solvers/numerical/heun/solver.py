"""Heun's Method (Improved Euler / Explicit Trapezoidal) ODE solver."""

import numpy as np

NAME = "Heun's Method"
KEY = "heun"
DESCRIPTION = "Second-order predictor-corrector method. Uses an Euler predictor and trapezoidal corrector."
IS_IMPLEMENTED = True


def solve(ode_func, t_span, y0, args, dt):
    """Solve the ODE system using Heun's method.

    Algorithm (2-stage explicit Runge-Kutta):
        k1 = f(t_n, y_n)                   — predictor (Euler slope)
        k2 = f(t_n + h, y_n + h * k1)      — corrector slope
        y_{n+1} = y_n + (h / 2) * (k1 + k2)

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

    for i in range(n_steps):
        h = t[i + 1] - t[i]
        y_n = y[:, i]

        # Stage 1: predictor (Euler slope)
        k1 = np.array(ode_func(t[i], y_n, *args))
        y_pred = y_n + h * k1

        # Stage 2: corrector slope at predicted point
        k2 = np.array(ode_func(t[i] + h, y_pred, *args))

        # Update: average the two slopes
        y[:, i + 1] = y_n + (h / 2.0) * (k1 + k2)

    return t, y
