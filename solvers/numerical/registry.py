"""Registry of numerical ODE solvers.

This keeps the solver list in a dedicated module so the Streamlit app can
import it without relying on package-level name resolution during reloads.
"""

from solvers.numerical import euler as euler
from solvers.numerical import backward_euler as backward_euler
from solvers.numerical import runge_kutta as runge_kutta
from solvers.numerical import bdf as bdf
from solvers.numerical import adams_bashforth as adams_bashforth
from solvers.numerical import heun as heun

ODE_SOLVERS = [euler, backward_euler, runge_kutta, bdf, adams_bashforth, heun]
