"""
Newton-Raphson solver for finding steady states of the gene regulatory network.

Uses numerical Jacobian computation (central differences) and falls back
to a pseudo-inverse when the Jacobian is singular.
"""

import warnings

import numpy as np

from solvers.core import algebraic_system


def compute_jacobian(func, vars, epsilon=1e-5):
    """Compute the Jacobian matrix using central finite differences.

    Parameters
    ----------
    func : callable
        Vector-valued function F(x) -> array of shape (n,).
    vars : array-like
        Point at which to evaluate the Jacobian.
    epsilon : float
        Step size for the finite-difference stencil.

    Returns
    -------
    numpy.ndarray
        Jacobian matrix of shape (n, n).
    """
    n_vars = len(vars)
    jacobian = np.zeros((n_vars, n_vars))
    for i in range(n_vars):
        v_plus = np.copy(vars)
        v_minus = np.copy(vars)
        v_plus[i] += epsilon
        v_minus[i] -= epsilon
        jacobian[:, i] = (func(v_plus) - func(v_minus)) / (2 * epsilon)
    return jacobian


def solve_newton(initial_guess, p, n, m, tolerance=1e-6, max_iter=50):
    """Find a steady state of the ODE system using Newton-Raphson iteration.

    Parameters
    ----------
    initial_guess : array-like
        Starting point [G0, P0].
    p : dict
        Model parameters.
    n, m : float
        Hill coefficients.
    tolerance : float
        Convergence threshold on the L2 norm of F(x).
    max_iter : int
        Maximum number of Newton iterations.

    Returns
    -------
    v : numpy.ndarray
        Approximate root [G*, P*].
    history : numpy.ndarray
        Array of shape (k, 2) recording each iterate.
    errors : list of float
        L2 norm of F(x) at each iteration.
    """
    v = np.array(initial_guess, dtype=float)
    history = [v.copy()]
    errors = []
    for i in range(max_iter):
        F_val = algebraic_system(v, p, n, m)
        err_norm = np.linalg.norm(F_val)
        errors.append(err_norm)
        if err_norm < tolerance:
            break
        J = compute_jacobian(lambda x: algebraic_system(x, p, n, m), v)
        # Add small damping or pseudo-inverse if singular, though standard inverse is usually fine
        try:
            delta = np.linalg.solve(J, -F_val)
        except np.linalg.LinAlgError:
            warnings.warn(
                "Jacobian is singular at iteration %d, using pseudo-inverse." % i,
                RuntimeWarning,
            )
            delta = np.linalg.pinv(J).dot(-F_val)
        v = v + delta
        history.append(v.copy())
    return v, np.array(history), errors
