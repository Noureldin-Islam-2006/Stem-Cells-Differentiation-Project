"""Numerical solvers sub-package.

Provides a registry of all ODE solvers (ODE_SOLVERS) and the Newton-Raphson
root finder. To add a new ODE method, create a new sub-package under
``solvers/numerical/<name>/`` with the standard interface and append the
module to ODE_SOLVERS below.
"""

from solvers.numerical.newton.solver import compute_jacobian, solve_newton
from solvers.numerical.registry import ODE_SOLVERS

__all__ = [
    "compute_jacobian",
    "solve_newton",
    "ODE_SOLVERS",
]
