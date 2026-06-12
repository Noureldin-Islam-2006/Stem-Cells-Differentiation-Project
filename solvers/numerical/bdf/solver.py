"""BDF ODE solver."""

import numpy as np
from scipy.optimize import fsolve

NAME = "BDF"
KEY = "bdf"
DESCRIPTION = "Implicit multistep method. Uses a fixed-step BDF-2 corrector with backward-Euler startup."
IS_IMPLEMENTED = True


def _finite_difference_jacobian(residual_func, y, epsilon=1e-6):
    y = np.asarray(y, dtype=float)
    n_vars = len(y)
    jacobian = np.zeros((n_vars, n_vars), dtype=float)
    for idx in range(n_vars):
        y_plus = y.copy()
        y_minus = y.copy()
        y_plus[idx] += epsilon
        y_minus[idx] -= epsilon
        jacobian[:, idx] = (residual_func(y_plus) - residual_func(y_minus)) / (2 * epsilon)
    return jacobian


def solve(ode_func, t_span, y0, args, dt):
    """Solve an IVP using a fixed-step BDF-2 method.

    Parameters
    ----------
    ode_func : callable
        The ODE right-hand side f(t, y, *args).
    t_span : tuple
        (t_start, t_end).
    y0 : array-like
        Initial conditions.
    args : tuple
        Extra arguments forwarded to ode_func.
    dt : float
        Fixed step size.

    Returns
    -------
    t : numpy.ndarray, shape (n_steps,)
    y : numpy.ndarray, shape (n_vars, n_steps)
    """
    t_start, t_end = t_span
    if dt <= 0:
        raise ValueError("Step size dt must be greater than zero.")
    if t_end < t_start:
        raise ValueError("Final time must be greater than or equal to initial time.")

    y0 = np.asarray(y0, dtype=float)
    n_vars = len(y0)
    n_steps = int(np.ceil((t_end - t_start) / dt))
    t = np.linspace(t_start, t_end, n_steps + 1)
    y = np.zeros((n_vars, len(t)), dtype=float)
    y[:, 0] = y0

    if n_steps == 0:
        return t, y

    if n_steps >= 1:
        t1 = t[1]
        step0 = float(t1 - t[0])
        y_guess = y[:, 0] + step0 * np.asarray(ode_func(t[0], y[:, 0], *args), dtype=float)
        y[:, 1] = fsolve(
            lambda y_new: y_new - y[:, 0] - step0 * np.asarray(ode_func(t1, y_new, *args), dtype=float),
            y_guess,
        )

    if n_steps == 1:
        return t, y

    # BDF-2 coefficients: 3/2 y_{n+1} - 2 y_n + 1/2 y_{n-1} = h f(t_{n+1}, y_{n+1})
    alpha0, alpha1, alpha2 = 3.0 / 2.0, -2.0, 1.0 / 2.0

    for idx in range(1, n_steps):
        t_next = t[idx + 1]
        h_actual = float(t_next - t[idx])
        y_prev = y[:, idx - 1]
        y_curr = y[:, idx]

        def residual(y_new):
            rhs = np.asarray(ode_func(t_next, y_new, *args), dtype=float)
            return alpha0 * y_new + alpha1 * y_curr + alpha2 * y_prev - h_actual * rhs

        def jacobian(y_new):
            return _finite_difference_jacobian(residual, y_new)

        y_guess = y_curr + (y_curr - y_prev)
        y[:, idx + 1] = fsolve(residual, y_guess, fprime=jacobian)

    return t, y
