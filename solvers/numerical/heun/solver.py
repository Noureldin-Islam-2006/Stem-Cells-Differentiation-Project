"""Heun's Method (Improved Euler / Explicit Trapezoidal) ODE solver."""

NAME = "Heun's Method"
KEY = "heun"
DESCRIPTION = "Second-order predictor-corrector method. Uses an Euler predictor and trapezoidal corrector."
IS_IMPLEMENTED = False


def solve(ode_func, t_span, y0, args, dt):
    """Solve the ODE system using Heun's method.

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
