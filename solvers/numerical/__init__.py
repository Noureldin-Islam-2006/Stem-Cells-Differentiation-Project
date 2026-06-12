"""Numerical solvers sub-package.

Provides a registry of all ODE solvers (ODE_SOLVERS) and the Newton-Raphson
root finder. To add a new ODE method, create a new sub-package under
``solvers/numerical/<name>/`` with the standard interface and append the
module to ODE_SOLVERS below.
"""

from solvers.numerical.newton.solver import compute_jacobian, solve_newton

from solvers.numerical import euler as euler
from solvers.numerical import backward_euler as backward_euler
from solvers.numerical import runge_kutta as runge_kutta
from solvers.numerical import bdf as bdf
from solvers.numerical import adams_bashforth as adams_bashforth
from solvers.numerical import heun as heun

# Ordered list of ODE solver modules — the UI iterates over this.
ODE_SOLVERS = [euler, backward_euler, runge_kutta, bdf, adams_bashforth, heun]

__all__ = [
    "compute_jacobian",
    "solve_newton",
    "ODE_SOLVERS",
]
