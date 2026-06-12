"""Classical 4th-order Runge-Kutta (RK4) ODE solver."""

import numpy as np

NAME = "Runge-Kutta (RK4)"
KEY = "runge_kutta"
DESCRIPTION = "Fourth-order explicit method. Uses four stage evaluations per step for high accuracy."
IS_IMPLEMENTED = True


def rk4_step(ode_func, t, y, h, args):
    y = np.asarray(y, dtype=float)
    k1 = np.asarray(ode_func(t, y, *args), dtype=float)
    k2 = np.asarray(ode_func(t + h / 2, y + h * k1 / 2, *args), dtype=float)
    k3 = np.asarray(ode_func(t + h / 2, y + h * k2 / 2, *args), dtype=float)
    k4 = np.asarray(ode_func(t + h, y + h * k3, *args), dtype=float)
    return y + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6


def solve(ode_func, t_span, y0, args, dt):
    """Solve the ODE system using RK4.

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
    if dt <= 0:
        raise ValueError("Step size dt must be greater than zero.")
    if t_end < t_start:
        raise ValueError("Final time must be greater than or equal to initial time.")

    n_steps = int(np.ceil((t_end - t_start) / dt))
    t = np.linspace(t_start, t_end, n_steps + 1)
    y = np.zeros((len(y0), len(t)), dtype=float)
    y[:, 0] = np.asarray(y0, dtype=float)

    for i in range(n_steps):
        step = float(t[i + 1] - t[i])
        y[:, i + 1] = rk4_step(ode_func, t[i], y[:, i], step, args)

    return t, y
