# PU.1/GATA-1 Gene Regulatory Network Simulation

An interactive Streamlit application for simulating the **PU.1/GATA-1 gene regulatory network** that governs hematopoietic stem cell differentiation. The project implements and compares multiple numerical ODE solvers and a Physics-Informed Neural Network (PINN), with built-in convergence analysis tools.

Based on the mathematical model from **Schiesser (2014), Chapter 5** — equations 5.1a and 5.1b.

---

## The Model

The system describes two competing transcription factors — **GATA-1 (G)** and **PU.1 (P)** — whose mutual inhibition and self-activation create a bistable switch that drives cell fate decisions:

$$\frac{dG}{dt} = \frac{a_1 G^n}{\theta_{a1}^n + G^n} + \frac{b_1 \theta_{b1}^m}{\theta_{b1}^m + G^m P^m} - k_1 G$$

$$\frac{dP}{dt} = \frac{a_2 P^n}{\theta_{a2}^n + P^n} + \frac{b_2 \theta_{b2}^m}{\theta_{b2}^m + G^m P^m} - k_2 P$$

Each equation has three terms:
- **Auto-activation** — positive feedback via Hill function
- **Basal production / cross-inhibition** — mutual suppression between G and P
- **Degradation** — linear decay

---

## Features

### System Dynamics
Solve the ODE system and visualize protein concentrations over time using any of the available numerical methods:

| Method | Type | Order | Description |
|--------|------|-------|-------------|
| **Reference (SciPy)** | Adaptive | Variable | LSODA with automatic stiffness detection |
| **Euler's Method** | Explicit | 1st | Forward Euler — simplest time-stepping scheme |
| **Backward Euler** | Implicit | 1st | Implicit Euler with Newton iteration (via `fsolve`) |
| **Runge-Kutta (RK4)** | Explicit | 4th | Classical 4-stage method |
| **BDF** | Implicit | Multi-step | Backward differentiation formula for stiff systems |
| **Adams-Bashforth** | Explicit | Multi-step | Uses previous evaluations to extrapolate |
| **Heun's Method** | Explicit | 2nd | Predictor-corrector (improved Euler) |

Each method has its own tab with configurable step size and automatic overlay of the SciPy reference solution.

### Phase Plane & Steady States
- **Nullcline visualization** — curves where dG/dt = 0 and dP/dt = 0
- **Vector field** — streamlines showing system trajectories
- **Newton-Raphson root finder** — iteratively locates steady states with visual path tracking

### Convergence Analysis
Compare solver accuracy with two configurable error plots:
- **Step Size Error** — max L2 error vs. step size *h* (sweep multiple values)
- **Iteration Error** — per-step L2 error vs. time for a given *h*
- **Newton-Raphson** — residual norm per iteration with adjustable tolerance and max iterations

All plots use linear scale. Step sizes, tolerances, and iteration limits are adjustable through the UI.

### Physics-Informed Neural Network (PINN)
Train a neural network to learn G(t) and P(t) using the ODEs as physics constraints — no labelled data required:
- Live training visualization with loss curves and prediction updates
- Configurable architecture (layers, neurons, learning rate, epochs)
- Comparison against the ODE reference solution
- Supports GPU acceleration when available

### Parameter Presets
Two built-in presets from the textbook:
- **Case 1 (Symmetric)** — stable progenitor state, both proteins converge to equal concentrations
- **Case 2 (Asymmetric)** — bistable switch, small perturbations drive differentiation toward one lineage

All 12 model parameters are individually adjustable via the sidebar.

---

## Project Structure

```
├── app.py                             # Streamlit UI (imports from solvers/)
├── requirements.txt
├── CONTRIBUTING.md                    # Guide for adding new solvers
│
└── solvers/
    ├── core.py                        # Shared ODE system equations
    │
    ├── numerical/
    │   ├── __init__.py                # Solver registry (ODE_SOLVERS list)
    │   ├── newton/solver.py           # Newton-Raphson root finder
    │   ├── euler/solver.py            # Forward Euler
    │   ├── backward_euler/solver.py   # Implicit Euler (fsolve)
    │   ├── runge_kutta/solver.py      # Classical RK4
    │   ├── bdf/solver.py              # Backward Differentiation Formula
    │   ├── adams_bashforth/solver.py  # Adams-Bashforth multi-step
    │   └── heun/solver.py             # Heun's method (improved Euler)
    │
    └── ml/
        └── pinn/
            ├── model.py               # PINN architecture (nn.Module)
            ├── loss.py                # Physics-informed loss functions
            └── utils.py               # Evaluation & visualization
```

### Architecture

The app is designed around a **solver registry** pattern. All ODE solvers expose the same interface:

```python
def solve(ode_func, t_span, y0, args, dt) -> (t, y)
```

The UI iterates over the `ODE_SOLVERS` list in `solvers/numerical/__init__.py` to generate tabs and convergence analysis automatically. Adding a new solver requires zero changes to `app.py` — see [CONTRIBUTING.md](CONTRIBUTING.md).

---

## Getting Started

### Prerequisites

- Python 3.9+
- pip

### Installation

```bash
git clone https://github.com/Noureldin-Islam-2006/Stem-Cells-Differentiation-Project.git
cd Stem-Cells-Differentiation-Project
pip install -r requirements.txt
```

### Running

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on adding new numerical or ML solvers. The short version:

1. Implement `solve()` in `solvers/numerical/<method>/solver.py`
2. Set `IS_IMPLEMENTED = True`
3. The UI picks it up automatically

---

## References

- Schiesser, W. E. (2014). *Differential Equation Analysis in Biomedical Science and Engineering: Ordinary Differential Equation Applications with R*. Chapter 5: Stem Cell Differentiation.
