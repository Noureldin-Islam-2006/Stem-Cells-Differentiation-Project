"""Forward Euler ODE solver."""

import numpy as np

NAME = "Euler's Method"
KEY = "euler"
DESCRIPTION = "First-order explicit method. Approximates y(t+h) ≈ y(t) + h·f(t, y)."
IS_IMPLEMENTED = True


def solve(ode_func, t_span, y0, args, dt):
    """Solve the ODE system using forward Euler.

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
    y = np.zeros((len(y0), len(t)))
    y[:, 0] = y0

    for i in range(n_steps):
        dydt = ode_func(t[i], y[:, i], *args)
        y[:, i + 1] = y[:, i] + dt * np.array(dydt)

    return t, y
