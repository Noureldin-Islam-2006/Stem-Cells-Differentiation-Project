"""Classical 4th-order Runge-Kutta (RK4) ODE solver."""

NAME = "Runge-Kutta (RK4)"
KEY = "runge_kutta"
DESCRIPTION = "Fourth-order explicit method. Uses four stage evaluations per step for high accuracy."
IS_IMPLEMENTED = False


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
    raise NotImplementedError(f"{NAME} is not yet implemented.")
