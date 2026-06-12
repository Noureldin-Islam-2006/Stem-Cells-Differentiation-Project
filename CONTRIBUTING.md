# Contributing

This guide explains how to add new solvers to the simulation. The project is designed so that **you only need to touch files inside `solvers/`** — the Streamlit UI will pick up your changes automatically.

---

## Project Structure

```
solvers/
├── core.py                        # Shared ODE system (used by all solvers)
├── numerical/
│   ├── __init__.py                # ← Solver registry (ODE_SOLVERS list)
│   ├── newton/                    # Root-finder (steady-state analysis)
│   ├── euler/                     # Placeholder
│   ├── backward_euler/            # Placeholder
│   ├── runge_kutta/               # Placeholder
│   ├── bdf/                       # Placeholder
│   ├── adams_bashforth/           # Placeholder
│   └── heun/                      # Placeholder
└── ml/
    └── pinn/                      # Physics-Informed Neural Network
```

---

## Adding a Numerical ODE Solver

### 1. Implement the `solve()` function

Open the placeholder file for the method you want to implement, e.g. `solvers/numerical/euler/solver.py`, and fill in the `solve()` function:

```python
"""Forward Euler ODE solver."""

import numpy as np

NAME = "Euler's Method"
KEY = "euler"
DESCRIPTION = "First-order explicit method. Approximates y(t+h) ≈ y(t) + h·f(t, y)."
IS_IMPLEMENTED = True  # ← Flip this to True


def solve(ode_func, t_span, y0, args, dt):
    t_start, t_end = t_span
    n_steps = int(np.ceil((t_end - t_start) / dt))
    t = np.linspace(t_start, t_end, n_steps + 1)
    y = np.zeros((len(y0), len(t)))
    y[:, 0] = y0

    for i in range(n_steps):
        dydt = ode_func(t[i], y[:, i], *args)
        y[:, i + 1] = y[:, i] + dt * np.array(dydt)

    return t, y
```

That's it. The UI will now show your solver's results in the **System Dynamics** tab and make it available in the **Convergence Analysis** tab.

### 2. Set `IS_IMPLEMENTED = True`

This flag controls everything:

| `IS_IMPLEMENTED` | System Dynamics tab | Convergence Analysis |
|---|---|---|
| `False` | Shows "🚧 not yet implemented" banner | Method hidden from dropdown |
| `True` | Shows dt control + solution plot | Step-size & iteration error available |

### 3. (Optional) Register a brand-new method

If you're adding a method that doesn't have a placeholder yet:

1. **Create the directory** `solvers/numerical/<your_method>/`
2. **Create `solver.py`** with the standard interface (see template below)
3. **Create `__init__.py`** that re-exports the module attributes:
   ```python
   """Your method solver package."""
   from solvers.numerical.<your_method>.solver import NAME, KEY, DESCRIPTION, IS_IMPLEMENTED, solve
   ```
4. **Register it** in `solvers/numerical/__init__.py`:
   ```python
   from solvers.numerical import <your_method> as <your_method>

   ODE_SOLVERS = [euler, backward_euler, ..., <your_method>]  # append here
   ```

---

## Solver Interface Reference

Every ODE solver module **must** expose these five attributes:

```python
NAME: str             # Human-readable name shown in UI tabs
KEY: str              # Unique snake_case identifier (used for widget keys)
DESCRIPTION: str      # One-line description shown under the tab header
IS_IMPLEMENTED: bool  # Set to True when solve() is ready

def solve(ode_func, t_span, y0, args, dt):
    """
    Parameters
    ----------
    ode_func : callable
        The ODE right-hand side: f(t, y, *args) → list of dy/dt.
        This is solvers.core.ode_system — you receive it, don't import it.
    t_span : tuple of float
        (t_start, t_end).
    y0 : list or array
        Initial conditions [G0, P0].
    args : tuple
        Extra arguments forwarded to ode_func. Unpack with *args:
            ode_func(t, y, *args)
    dt : float
        Step size chosen by the user in the UI.

    Returns
    -------
    t : numpy.ndarray, shape (n_steps,)
        Time points (must start at t_start and end at or near t_end).
    y : numpy.ndarray, shape (n_vars, n_steps)
        Solution array. Row 0 = GATA-1 (G), Row 1 = PU.1 (P).
        This matches the SciPy convention (sol.y).
    """
```

> **Important:** The return shape is `(n_vars, n_steps)`, not `(n_steps, n_vars)`. This matches SciPy's `solve_ivp` output format so the UI plotting code works uniformly.

---

## The Shared ODE System

All solvers use the same equations defined in `solvers/core.py`. You should **not** redefine the ODE — instead, call the `ode_func` that is passed into your `solve()` function:

```python
# Inside your solver's time-stepping loop:
dydt = ode_func(t[i], y[:, i], *args)   # returns [dG/dt, dP/dt]
```

The `args` tuple contains `(p, n, m)` where:
- `p` — dict of model parameters (`a1`, `a2`, `b1`, `b2`, `tha1`, `tha2`, `thb1`, `thb2`, `k1`, `k2`)
- `n` — auto-activation Hill coefficient
- `m` — cross-inhibition Hill coefficient

---

## Adding an ML Solver

ML solvers live under `solvers/ml/`. The existing PINN is structured as:

```
solvers/ml/pinn/
├── model.py    # nn.Module architecture
├── loss.py     # Physics-informed loss functions
└── utils.py    # Evaluation & visualization helpers
```

To add a new ML approach (e.g., a Neural ODE, DeepONet, etc.):

1. Create `solvers/ml/<your_method>/` with the relevant files
2. Add a new tab or sub-tab in `app.py` under the Machine Learning section
3. ML solvers have more flexibility in their interface since training workflows vary

---

## Convergence Analysis

When you flip `IS_IMPLEMENTED = True`, two convergence analyses become available automatically:

### Step Size Error
Runs your solver with multiple step sizes and plots the max L2 error against the SciPy reference. This is useful for verifying the order of your method (e.g., Euler should show linear convergence, RK4 should show 4th-order).

### Iteration Error
Runs your solver with a single step size and plots the L2 error at each time step. This shows how error accumulates over time.

Both plots use **linear scale** (no log). All parameters (step sizes, tolerances) are configurable through the UI.

---

## Quick Checklist

- [ ] Implement `solve()` in `solvers/numerical/<method>/solver.py`
- [ ] Set `IS_IMPLEMENTED = True`
- [ ] Return `(t, y)` with `y` shape `(n_vars, n_steps)`
- [ ] Call `ode_func(t, y, *args)` — don't hardcode the ODE
- [ ] Test by running `streamlit run app.py` and checking your method's tab

---

## Running the App

```bash
pip install -r requirements.txt
streamlit run app.py
```
