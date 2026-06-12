"""Forward Euler ODE solver."""

import numpy as np

NAME = "Euler's Method"
KEY = "euler"
DESCRIPTION = "First-order explicit method. Approximates y(t+h) ≈ y(t) + h·f(t, y)."
IS_IMPLEMENTED = False


def solve(ode_func, t_span, y0, args, dt):
    raise NotImplementedError(f"{NAME} is not yet implemented.")