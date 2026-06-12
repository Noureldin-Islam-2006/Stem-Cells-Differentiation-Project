"""Backward (Implicit) Euler ODE solver."""

import numpy as np

import numpy as np
from scipy.optimize import fsolve

NAME = "Backward Euler"
KEY = "backward_euler"
DESCRIPTION = "First-order implicit method. Solves y(t+h) = y(t) + h·f(t+h, y(t+h)) via Newton iteration at each step."
IS_IMPLEMENTED = True


def solve(ode_func, t_span, y0, args, dt):
    """Solve the ODE system using backward Euler.

    At each step we solve the implicit equation
        y_{n+1} = y_n + h * f(t_{n+1}, y_{n+1})
    using scipy.optimize.fsolve with a forward-Euler predictor as the
    initial guess.

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
        y_prev = y[:, i]

        # Implicit residual: y_next - y_prev - h * f(t_next, y_next) = 0
        def residual(y_next):
            dydt = np.array(ode_func(t[i + 1], y_next, *args))
            return y_next - y_prev - dt * dydt

        # Use a forward-Euler step as the initial guess for fsolve
        dydt_prev = np.array(ode_func(t[i], y_prev, *args))
        y_guess = y_prev + dt * dydt_prev

        sol, _info, ier, _msg = fsolve(residual, y_guess, full_output=True)
        y[:, i + 1] = sol

    return t, y
