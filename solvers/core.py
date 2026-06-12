"""
Shared ODE system for the PU.1/GATA-1 Gene Regulatory Network.

This module defines the governing equations used by all solvers
(numerical and machine learning).
"""

import numpy as np


def calculate_terms(G, P, p, n, m):
    """Break the ODE right-hand side into individual biological terms.

    Returns
    -------
    Gterm1 : auto-activation of GATA-1
    Gterm2 : basal production / cross-inhibition of GATA-1
    Gterm3 : degradation of GATA-1
    Pterm1 : auto-activation of PU.1
    Pterm2 : basal production / cross-inhibition of PU.1
    Pterm3 : degradation of PU.1
    """
    Gterm1 = p['a1'] * (G**n) / (p['tha1']**n + G**n)
    Gterm2 = p['b1'] * (p['thb1']**m) / (p['thb1']**m + (G**m) * (P**m))
    Gterm3 = -p['k1'] * G
    Pterm1 = p['a2'] * (P**n) / (p['tha2']**n + P**n)
    Pterm2 = p['b2'] * (p['thb2']**m) / (p['thb2']**m + (G**m) * (P**m))
    Pterm3 = -p['k2'] * P
    return Gterm1, Gterm2, Gterm3, Pterm1, Pterm2, Pterm3


def ode_system(t, vars, p, n, m):
    """Right-hand side of the PU.1/GATA-1 ODE system.

    Parameters
    ----------
    t : float
        Current time (unused for autonomous systems, kept for solver APIs).
    vars : array-like
        [G, P] — current concentrations of GATA-1 and PU.1.
    p : dict
        Model parameters (a1, a2, b1, b2, tha1, tha2, thb1, thb2, k1, k2).
    n : float
        Auto-activation Hill coefficient.
    m : float
        Cross-inhibition Hill coefficient.

    Returns
    -------
    list of float
        [dG/dt, dP/dt]
    """
    G, P = vars
    g1, g2, g3, p1, p2, p3 = calculate_terms(G, P, p, n, m)
    return [g1 + g2 + g3, p1 + p2 + p3]


def algebraic_system(vars, p, n, m):
    """Steady-state form of the ODE (dG/dt = 0, dP/dt = 0).

    Used by root-finding solvers like Newton-Raphson.
    """
    return np.array(ode_system(0, vars, p, n, m))
