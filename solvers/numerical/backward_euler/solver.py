"""Backward (Implicit) Euler ODE solver."""

NAME = "Backward Euler"
KEY = "backward_euler"
DESCRIPTION = "First-order implicit method. Solves y(t+h) = y(t) + h·f(t+h, y(t+h)) via Newton iteration at each step."
IS_IMPLEMENTED = False


def solve(ode_func, t_span, y0, args, dt):
    """Solve the ODE system using backward Euler.

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
    raise NotImplementedError(f"{NAME} is not yet implemented.")
