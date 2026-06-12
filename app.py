import warnings

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch

# --- Solver Imports ---
from solvers.core import ode_system, calculate_terms
from solvers.numerical.newton import solve_newton
from solvers.numerical.registry import ODE_SOLVERS
from solvers.ml.pinn import (
    PINN,
    calculate_pinn_loss,
    evaluate_pinn,
    create_live_training_figure,
)
from solvers.ml.lno import (
    LaplaceNeuralOperator,
    generate_training_data,
    prepare_dataloaders,
    train_one_epoch,
    evaluate as lno_evaluate,
    predict_sample,
    create_lno_training_figure,
)

# --- Streamlit UI Architecture ---

st.set_page_config(page_title="PU.1/GATA-1 Gene Regulatory Network Simulation", layout="wide")

st.title("PU.1/GATA-1 Gene Regulatory Network Simulation")
st.markdown("This interactive simulation models the PU.1/GATA-1 Gene Regulatory Network, which dictates hematopoietic stem cell differentiation.")

# 1. Sidebar (Parameter Controls)
st.sidebar.header("Model Parameters")

preset = st.sidebar.selectbox("Preset", ["Custom", "Case 1 (Symmetric/Undifferentiated)", "Case 2 (Asymmetric/Bistable Switch)"])

# Define default params based on preset
if preset == "Case 1 (Symmetric/Undifferentiated)":
    params_def = {'a1': 1.0, 'a2': 1.0, 'b1': 1.0, 'b2': 1.0, 'tha1': 1.0, 'tha2': 1.0, 'thb1': 1.0, 'thb2': 1.0, 'k1': 1.0, 'k2': 1.0, 'n': 4.0, 'm': 1.0}
elif preset == "Case 2 (Asymmetric/Bistable Switch)":
    params_def = {'a1': 5.0, 'a2': 10.0, 'b1': 1.0, 'b2': 1.0, 'tha1': 0.5, 'tha2': 0.5, 'thb1': 0.071, 'thb2': 0.071, 'k1': 1.0, 'k2': 1.0, 'n': 4.0, 'm': 1.0}
else:
    # Defaults for custom can be Case 1
    params_def = {'a1': 1.0, 'a2': 1.0, 'b1': 1.0, 'b2': 1.0, 'tha1': 1.0, 'tha2': 1.0, 'thb1': 1.0, 'thb2': 1.0, 'k1': 1.0, 'k2': 1.0, 'n': 4.0, 'm': 1.0}

p = {}
col_s1, col_s2 = st.sidebar.columns(2)
with col_s1:
    p['a1'] = st.number_input("a1", value=float(params_def['a1']), format="%.3f")
    p['b1'] = st.number_input("b1", value=float(params_def['b1']), format="%.3f")
    p['tha1'] = st.number_input("tha1", value=float(params_def['tha1']), format="%.3f")
    p['thb1'] = st.number_input("thb1", value=float(params_def['thb1']), format="%.3f")
    p['k1'] = st.number_input("k1", value=float(params_def['k1']), format="%.3f")
with col_s2:
    p['a2'] = st.number_input("a2", value=float(params_def['a2']), format="%.3f")
    p['b2'] = st.number_input("b2", value=float(params_def['b2']), format="%.3f")
    p['tha2'] = st.number_input("tha2", value=float(params_def['tha2']), format="%.3f")
    p['thb2'] = st.number_input("thb2", value=float(params_def['thb2']), format="%.3f")
    p['k2'] = st.number_input("k2", value=float(params_def['k2']), format="%.3f")

n = st.sidebar.number_input("n (Auto-activation Hill coeff)", value=float(params_def['n']), format="%.3f")
m = st.sidebar.number_input("m (Cross-inhibition Hill coeff)", value=float(params_def['m']), format="%.3f")

st.sidebar.subheader("Initial Conditions & Time")
G0 = st.sidebar.number_input("Initial GATA-1 (G0)", value=0.0, format="%.3f")
P0 = st.sidebar.number_input("Initial PU.1 (P0)", value=0.0, format="%.3f")
t_end = st.sidebar.number_input("Time Span (t_end)", value=100.0, format="%.1f")

# 2. Reference solution (used by multiple tabs)
t_span = (0, t_end)
t_eval = np.linspace(0, t_end, max(1000, int(t_end*10)))
sol = solve_ivp(ode_system, t_span, [G0, P0], args=(p, n, m), t_eval=t_eval, method='LSODA')

# 3. Main Panel (Tabs)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["System Dynamics", "Phase Plane & Steady States", "Convergence Analysis", "Machine Learning", "Benchmarking"])

with tab1:
    st.header("System Dynamics")

    # Build sub-tab names from the solver registry
    method_names = ["Reference (SciPy)", "Book Reference (Schiesser)"] + [s.NAME for s in ODE_SOLVERS]
    method_tabs = st.tabs(method_names)

    # --- Reference (SciPy) sub-tab ---
    with method_tabs[0]:
        if sol.success:
            fig1, ax1 = plt.subplots(figsize=(10, 5))
            ax1.plot(sol.t, sol.y[0], label='GATA-1 (G)', color='red', linewidth=2)
            ax1.plot(sol.t, sol.y[1], label='PU.1 (P)', color='blue', linewidth=2)
            ax1.set_xlabel('Time')
            ax1.set_ylabel('Concentration')
            ax1.set_title('Protein Concentrations over Time (SciPy LSODA)')
            ax1.legend()
            ax1.grid(True)
            st.pyplot(fig1)
            plt.close(fig1)

            g1, g2, g3, p1, p2, p3 = calculate_terms(sol.y[0], sol.y[1], p, n, m)
            st.subheader("Component Breakdown")
            col1, col2 = st.columns(2)
            with col1:
                fig2, ax2 = plt.subplots(figsize=(6, 4))
                ax2.plot(sol.t, g1, label='Auto-activation', color='darkred', linestyle='--')
                ax2.plot(sol.t, g2, label='Basal/Cross-inhibition', color='salmon', linestyle='-.')
                ax2.plot(sol.t, g3, label='Degradation', color='black', linestyle=':')
                ax2.plot(sol.t, g1+g2+g3, label='Net dG/dt', color='red', linewidth=2)
                ax2.set_xlabel('Time')
                ax2.set_ylabel('Rate')
                ax2.set_title('GATA-1 Rate Terms')
                ax2.legend()
                ax2.grid(True)
                st.pyplot(fig2)
                plt.close(fig2)
            with col2:
                fig3, ax3 = plt.subplots(figsize=(6, 4))
                ax3.plot(sol.t, p1, label='Auto-activation', color='darkblue', linestyle='--')
                ax3.plot(sol.t, p2, label='Basal/Cross-inhibition', color='lightblue', linestyle='-.')
                ax3.plot(sol.t, p3, label='Degradation', color='black', linestyle=':')
                ax3.plot(sol.t, p1+p2+p3, label='Net dP/dt', color='blue', linewidth=2)
                ax3.set_xlabel('Time')
                ax3.set_ylabel('Rate')
                ax3.set_title('PU.1 Rate Terms')
                ax3.legend()
                ax3.grid(True)
                st.pyplot(fig3)
                plt.close(fig3)
        else:
            st.error("ODE Solver failed to converge.")

    # --- Book Reference (LSODA — Schiesser Ch. 5) sub-tab ---
    with method_tabs[1]:
        st.subheader("LSODA — Schiesser Ch. 5")
        st.markdown(
            "The solver used in the original R implementation from *Differential Equation "
            "Analysis in Biomedical Science and Engineering* (Schiesser, 2014). "
            "Python's `scipy.integrate.solve_ivp` with `method='LSODA'` is the equivalent "
            "of R's `lsodes`."
        )

        book_nout = st.number_input(
            "Output points", min_value=10, max_value=500, value=26, step=1, key="book_nout",
            help="The book uses 26 output points (nout=26).",
        )

        book_t_eval = np.linspace(0, t_end, int(book_nout))
        book_sol = solve_ivp(
            ode_system, t_span, [G0, P0],
            args=(p, n, m), t_eval=book_t_eval,
            method='LSODA', rtol=1e-8, atol=1e-8,
        )

        if not book_sol.success:
            st.error("LSODA solver failed to converge.")
        else:
            book_G = book_sol.y[0]
            book_P = book_sol.y[1]

            # Main plot — same style as other solver tabs
            fig_bk, ax_bk = plt.subplots(figsize=(10, 5))
            ax_bk.plot(book_t_eval, book_G, label='GATA-1 (G)', color='red', linewidth=2)
            ax_bk.plot(book_t_eval, book_P, label='PU.1 (P)', color='blue', linewidth=2)
            if sol.success:
                ax_bk.plot(sol.t, sol.y[0], 'r--', alpha=0.4, label='G ref (SciPy)')
                ax_bk.plot(sol.t, sol.y[1], 'b--', alpha=0.4, label='P ref (SciPy)')
            ax_bk.set_xlabel('Time')
            ax_bk.set_ylabel('Concentration')
            ax_bk.set_title(f'LSODA  ({int(book_nout)} output points)')
            ax_bk.legend()
            ax_bk.grid(True)
            st.pyplot(fig_bk)
            plt.close(fig_bk)

            # Compute derivatives at each output point (the book's approach)
            n_pts = len(book_t_eval)
            book_dG = np.zeros(n_pts)
            book_dP = np.zeros(n_pts)
            for j in range(n_pts):
                derivs = ode_system(book_t_eval[j], [book_G[j], book_P[j]], p, n, m)
                book_dG[j] = derivs[0]
                book_dP[j] = derivs[1]

            # 4-panel derivative plot (book's Figure 5.1 layout)
            st.subheader("Derivative Analysis (Book Layout)")
            fig_deriv, axes = plt.subplots(2, 2, figsize=(10, 7))

            axes[0, 0].plot(book_t_eval, book_G, 'r-', linewidth=2)
            axes[0, 0].set_xlabel("t")
            axes[0, 0].set_ylabel("G(t)")
            axes[0, 0].set_title("G(t), LSODA")
            axes[0, 0].grid(True, ls="--", alpha=0.5)

            axes[0, 1].plot(book_t_eval, book_P, 'b-', linewidth=2)
            axes[0, 1].set_xlabel("t")
            axes[0, 1].set_ylabel("P(t)")
            axes[0, 1].set_title("P(t), LSODA")
            axes[0, 1].grid(True, ls="--", alpha=0.5)

            axes[1, 0].plot(book_t_eval, book_dG, 'r-', linewidth=2)
            axes[1, 0].set_xlabel("t")
            axes[1, 0].set_ylabel("dG(t)/dt")
            axes[1, 0].set_title("dG(t)/dt")
            axes[1, 0].grid(True, ls="--", alpha=0.5)

            axes[1, 1].plot(book_t_eval, book_dP, 'b-', linewidth=2)
            axes[1, 1].set_xlabel("t")
            axes[1, 1].set_ylabel("dP(t)/dt")
            axes[1, 1].set_title("dP(t)/dt")
            axes[1, 1].grid(True, ls="--", alpha=0.5)

            fig_deriv.tight_layout()
            st.pyplot(fig_deriv)
            plt.close(fig_deriv)

            # Tabular output
            with st.expander("Numerical Table", expanded=False):
                table_data = {
                    "t":      [f"{t:.2f}" for t in book_t_eval],
                    "G":      [f"{g:.3f}" for g in book_G],
                    "P":      [f"{pp:.3f}" for pp in book_P],
                    "dG/dt":  [f"{dg:.3f}" for dg in book_dG],
                    "dP/dt":  [f"{dp:.3f}" for dp in book_dP],
                }
                st.dataframe(pd.DataFrame(table_data), use_container_width=True, hide_index=True)


    # --- ODE method sub-tabs (driven by registry) ---
    for idx, solver_mod in enumerate(ODE_SOLVERS):
        with method_tabs[idx + 2]:
            st.subheader(solver_mod.NAME)
            st.markdown(solver_mod.DESCRIPTION)

            if not solver_mod.IS_IMPLEMENTED:
                st.info(
                    f"🚧 **{solver_mod.NAME}** is not yet implemented. "
                    f"Implement it in `solvers/numerical/{solver_mod.KEY}/solver.py`."
                )
            else:
                solver_dt = st.number_input(
                    "Step size (dt)",
                    min_value=0.001,
                    max_value=float(t_end),
                    value=min(0.1, float(t_end) / 10),
                    format="%.4f",
                    key=f"dt_{solver_mod.KEY}",
                )
                try:
                    t_sol, y_sol = solver_mod.solve(
                        ode_system, t_span, [G0, P0], (p, n, m), solver_dt
                    )
                    fig_s, ax_s = plt.subplots(figsize=(10, 5))
                    ax_s.plot(t_sol, y_sol[0], label='GATA-1 (G)', color='red', linewidth=2)
                    ax_s.plot(t_sol, y_sol[1], label='PU.1 (P)', color='blue', linewidth=2)
                    if sol.success:
                        ax_s.plot(sol.t, sol.y[0], 'r--', alpha=0.4, label='G ref (SciPy)')
                        ax_s.plot(sol.t, sol.y[1], 'b--', alpha=0.4, label='P ref (SciPy)')
                    ax_s.set_xlabel('Time')
                    ax_s.set_ylabel('Concentration')
                    ax_s.set_title(f'{solver_mod.NAME}  (dt = {solver_dt})')
                    ax_s.legend()
                    ax_s.grid(True)
                    st.pyplot(fig_s)
                    plt.close(fig_s)
                except Exception as exc:
                    st.error(f"Solver error: {exc}")

with tab2:
    st.header("Phase Plane & Steady States (Newton-Raphson)")
    
    with warnings.catch_warnings(record=True) as caught_warnings:
        warnings.simplefilter("always")
        root, history, errors = solve_newton([G0, P0], p, n, m)
    for w in caught_warnings:
        st.warning(str(w.message))
    
    # Heuristic for determining grid bounds
    max_val = max(2.0, G0 * 1.5, P0 * 1.5, root[0] * 1.5, root[1] * 1.5)
    # Increase range specifically for asymmetric case to show all roots
    if preset == "Case 2 (Asymmetric/Bistable Switch)":
        max_val = max(max_val, 15.0)
    elif preset == "Case 1 (Symmetric/Undifferentiated)":
        max_val = max(max_val, 3.0)
        
    g_vals = np.linspace(0, max_val, 200)
    p_vals = np.linspace(0, max_val, 200)
    G_grid, P_grid = np.meshgrid(g_vals, p_vals)
    
    u, v = np.zeros_like(G_grid), np.zeros_like(P_grid)
    for i in range(G_grid.shape[0]):
        for j in range(G_grid.shape[1]):
            res = ode_system(0, [G_grid[i,j], P_grid[i,j]], p, n, m)
            u[i,j] = res[0]
            v[i,j] = res[1]
            
    fig4, ax4 = plt.subplots(figsize=(10, 8))
    
    # Vector field streamplot
    speed = np.sqrt(u**2 + v**2)
    lw = 1.5 * speed / (speed.max() + 1e-6)
    ax4.streamplot(G_grid, P_grid, u, v, density=1.5, color='lightgray', linewidth=lw)
    
    # Nullclines
    ax4.contour(G_grid, P_grid, u, levels=[0], colors='red', linewidths=2, linestyles='--')
    ax4.contour(G_grid, P_grid, v, levels=[0], colors='blue', linewidths=2, linestyles='--')
    
    # Custom legend entries for contours
    ax4.plot([], [], 'r--', linewidth=2, label='dG/dt = 0 Nullcline')
    ax4.plot([], [], 'b--', linewidth=2, label='dP/dt = 0 Nullcline')
    
    # Newton-Raphson Tracking Path
    if len(history) > 0:
        ax4.plot(history[:, 0], history[:, 1], 'go-', label='Newton Path', markersize=4, linewidth=1.5)
        ax4.plot(history[0, 0], history[0, 1], 'yo', label='Start Guess', markersize=8)
        ax4.plot(root[0], root[1], 'k*', label='Final Root', markersize=12)
        
    ax4.set_xlabel('GATA-1 (G)')
    ax4.set_ylabel('PU.1 (P)')
    ax4.set_title('Phase Plane: Nullclines & Vector Field')
    ax4.legend(loc='upper right')
    ax4.set_xlim([0, max_val])
    ax4.set_ylim([0, max_val])
    ax4.grid(True)
    
    st.pyplot(fig4)
    plt.close(fig4)
    st.success(f"**Found Steady State (Root):** G = {root[0]:.4f}, P = {root[1]:.4f}")

with tab3:
    st.header("Convergence Analysis")

    # Build method list: Newton + implemented ODE solvers
    conv_methods = ["Newton-Raphson"] + [s.NAME for s in ODE_SOLVERS if s.IS_IMPLEMENTED]
    selected_conv = st.selectbox("Method to analyse", conv_methods, key="conv_method")

    if selected_conv == "Newton-Raphson":
        st.markdown("Convergence of Newton-Raphson root finder against the SciPy reference steady state.")

        if not sol.success:
            st.error("Reference solution failed — cannot compute errors.")
        else:
            err_col1, err_col2 = st.columns(2)

            # Compute the SciPy reference steady state (final value of long integration)
            G_ref_ss = sol.y[0, -1]
            P_ref_ss = sol.y[1, -1]
            ref_norm = np.sqrt(G_ref_ss**2 + P_ref_ss**2)
            if ref_norm < 1e-15:
                ref_norm = 1.0  # avoid division by zero

            # --- Tolerance sweep (analogous to step-size error) ---
            with err_col1:
                st.subheader("Tolerance Sweep")
                nr_tol_min_exp = st.number_input("Min tolerance (exponent)", min_value=-15, max_value=-1, value=-10, step=1, key="nr_tol_min")
                nr_tol_max_exp = st.number_input("Max tolerance (exponent)", min_value=-10, max_value=0, value=-1, step=1, key="nr_tol_max")
                nr_tol_count = st.number_input("Number of tolerances", min_value=2, max_value=20, value=8, step=1, key="nr_tol_count")

                tol_values = np.logspace(nr_tol_max_exp, nr_tol_min_exp, int(nr_tol_count))
                tol_errors = []
                for tol_val in tol_values:
                    with warnings.catch_warnings(record=True):
                        warnings.simplefilter("always")
                        root, _, _ = solve_newton([G0, P0], p, n, m, tolerance=float(tol_val), max_iter=200)
                    rel_err = np.sqrt((root[0] - G_ref_ss)**2 + (root[1] - P_ref_ss)**2) / ref_norm
                    tol_errors.append(rel_err)

                fig_ts, ax_ts = plt.subplots(figsize=(6, 4))
                ax_ts.plot(tol_values, tol_errors, 'rs-', linewidth=2)
                ax_ts.set_xlabel('Tolerance')
                ax_ts.set_ylabel('Relative L2 Error vs Reference')
                ax_ts.set_title('Error vs Tolerance')
                ax_ts.set_xscale('log')
                ax_ts.invert_xaxis()
                ax_ts.grid(True, ls="--")
                st.pyplot(fig_ts)
                plt.close(fig_ts)

            # --- Per-Iteration Error ---
            with err_col2:
                st.subheader("Iteration Error")
                nr_tol = st.number_input("Tolerance", min_value=1e-12, max_value=1.0, value=1e-6, format="%.1e", key="nr_tol")
                nr_max = st.number_input("Max iterations", min_value=1, max_value=500, value=50, step=1, key="nr_max")
                with warnings.catch_warnings(record=True) as cw:
                    warnings.simplefilter("always")
                    _, nr_history, _ = solve_newton([G0, P0], p, n, m, tolerance=nr_tol, max_iter=nr_max)
                for w in cw:
                    st.warning(str(w.message))

                if len(nr_history) > 1:
                    iter_rel_errs = []
                    for iterate in nr_history:
                        rel_err = np.sqrt((iterate[0] - G_ref_ss)**2 + (iterate[1] - P_ref_ss)**2) / ref_norm
                        iter_rel_errs.append(rel_err)

                    fig_it, ax_it = plt.subplots(figsize=(6, 4))
                    ax_it.plot(range(len(iter_rel_errs)), iter_rel_errs, 'b-o', linewidth=2)
                    ax_it.set_xlabel('Iteration')
                    ax_it.set_ylabel('Relative L2 Error vs Reference')
                    ax_it.set_title(f'Per-Iteration Error  (tol = {nr_tol:.0e})')
                    ax_it.grid(True, ls="--")
                    st.pyplot(fig_it)
                    plt.close(fig_it)
                else:
                    st.write("Initial guess is already at the root — no iterations needed.")

    else:
        # Find the selected solver module
        solver_mod = next(s for s in ODE_SOLVERS if s.NAME == selected_conv)
        st.markdown(f"Error analysis for **{solver_mod.NAME}** against the SciPy LSODA reference.")

        if not sol.success:
            st.error("Reference solution failed — cannot compute errors.")
        else:
            err_col1, err_col2 = st.columns(2)

            # --- Step Size Error ---
            with err_col1:
                st.subheader("Step Size Error")
                ss_min = st.number_input("Min step size", min_value=0.001, value=0.05, format="%.4f", key="ss_min")
                ss_max = st.number_input("Max step size", min_value=0.01, value=min(5.0, t_end / 2), format="%.4f", key="ss_max")
                ss_count = st.number_input("Number of step sizes", min_value=2, max_value=50, value=8, step=1, key="ss_count")

                dt_values = np.linspace(ss_min, ss_max, int(ss_count))
                ss_errors = []
                for dt_val in dt_values:
                    try:
                        t_s, y_s = solver_mod.solve(ode_system, t_span, [G0, P0], (p, n, m), float(dt_val))
                        G_interp = np.interp(t_s, sol.t, sol.y[0])
                        P_interp = np.interp(t_s, sol.t, sol.y[1])
                        ref_norms = np.sqrt(G_interp**2 + P_interp**2)
                        ref_norms = np.where(ref_norms < 1e-15, 1.0, ref_norms)
                        err = np.max(np.sqrt((y_s[0] - G_interp)**2 + (y_s[1] - P_interp)**2) / ref_norms)
                        ss_errors.append(err)
                    except Exception:
                        ss_errors.append(np.nan)

                fig_ss, ax_ss = plt.subplots(figsize=(6, 4))
                ax_ss.plot(dt_values, ss_errors, 'rs-', linewidth=2)
                ax_ss.set_xlabel('Step Size (h)')
                ax_ss.set_ylabel('Max Relative L2 Error')
                ax_ss.set_title('Step Size Error')
                ax_ss.grid(True, ls="--")
                st.pyplot(fig_ss)
                plt.close(fig_ss)

            # --- Iteration (Per-Step) Error ---
            with err_col2:
                st.subheader("Iteration Error")
                iter_dt = st.number_input(
                    "Step size for per-step analysis",
                    min_value=0.001, max_value=float(t_end),
                    value=min(0.1, float(t_end) / 10),
                    format="%.4f", key="iter_dt",
                )
                try:
                    t_it, y_it = solver_mod.solve(ode_system, t_span, [G0, P0], (p, n, m), float(iter_dt))
                    G_ref_it = np.interp(t_it, sol.t, sol.y[0])
                    P_ref_it = np.interp(t_it, sol.t, sol.y[1])
                    ref_norms_it = np.sqrt(G_ref_it**2 + P_ref_it**2)
                    ref_norms_it = np.where(ref_norms_it < 1e-15, 1.0, ref_norms_it)
                    iter_errs = np.sqrt((y_it[0] - G_ref_it)**2 + (y_it[1] - P_ref_it)**2) / ref_norms_it

                    fig_it, ax_it = plt.subplots(figsize=(6, 4))
                    ax_it.plot(t_it, iter_errs, 'b-', linewidth=2)
                    ax_it.set_xlabel('Time')
                    ax_it.set_ylabel('Relative L2 Error')
                    ax_it.set_title(f'Per-Step Error  (dt = {iter_dt})')
                    ax_it.grid(True, ls="--")
                    st.pyplot(fig_it)
                    plt.close(fig_it)
                except Exception as exc:
                    st.error(f"Solver error: {exc}")


with tab4:
    st.header("Machine Learning")
    ml_tab_pinn, ml_tab_lno = st.tabs(["Physics-Informed Neural Network (PINN)", "Laplace Neural Operator (LNO)"])

    with ml_tab_pinn:
        st.subheader("Physics-Informed Neural Network")
        st.markdown(
            "Train a neural network to predict **GATA-1** and **PU.1** while the "
            "gene-regulation ODEs act as its training constraints. The dashed curves "
            "are shown only as a reference and are not used as training labels."
        )

        if not sol.success:
            st.error("The reference ODE solution failed, so PINN comparison is unavailable.")
        elif t_end <= 0:
            st.error("Time Span must be greater than zero before training the PINN.")
        elif min(G0, P0) < 0:
            st.error("PINN training requires non-negative initial concentrations.")
        elif min(p['tha1'], p['tha2'], p['thb1'], p['thb2'], n, m) <= 0:
            st.error(
                "Hill thresholds and Hill coefficients must be greater than zero "
                "before training the PINN."
            )
        else:
            with st.expander("PINN Training Parameters", expanded=True):
                ml_col1, ml_col2, ml_col3 = st.columns(3)
                with ml_col1:
                    pinn_epochs = st.number_input(
                        "Epochs", min_value=1, max_value=50000, value=1000, step=100
                    )
                    pinn_learning_rate = st.number_input(
                        "Learning rate",
                        min_value=0.000001,
                        max_value=0.1,
                        value=0.01,
                        format="%.6f",
                    )
                    pinn_collocation = st.number_input(
                        "Collocation points",
                        min_value=10,
                        max_value=5000,
                        value=200,
                        step=10,
                    )
                with ml_col2:
                    pinn_layers = st.number_input(
                        "Hidden layers", min_value=1, max_value=10, value=3, step=1
                    )
                    pinn_neurons = st.number_input(
                        "Neurons per layer",
                        min_value=4,
                        max_value=512,
                        value=32,
                        step=4,
                    )
                    pinn_seed = st.number_input(
                        "Random seed", min_value=0, max_value=100000, value=42, step=1
                    )
                with ml_col3:
                    pinn_ic_weight = st.number_input(
                        "Initial-condition loss weight",
                        min_value=0.0,
                        max_value=10000.0,
                        value=10.0,
                        format="%.2f",
                    )
                    pinn_physics_weight = st.number_input(
                        "Physics loss weight",
                        min_value=0.0,
                        max_value=10000.0,
                        value=1.0,
                        format="%.2f",
                    )
                    pinn_update_every = st.number_input(
                        "Live update every N epochs",
                        min_value=1,
                        max_value=50000,
                        value=min(10, int(pinn_epochs)),
                        step=1,
                        help="Set to 1 to watch every epoch. Larger values train faster.",
                    )

            train_pinn = st.button(
                "Train PINN",
                type="primary",
                help="Training runs in this page and updates the charts as it learns.",
            )

            if train_pinn:
                torch.manual_seed(int(pinn_seed))
                np.random.seed(int(pinn_seed))
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

                model = PINN(
                    t_end=t_end,
                    hidden_layers=int(pinn_layers),
                    neurons=int(pinn_neurons),
                ).to(device)
                optimizer = torch.optim.Adam(
                    model.parameters(), lr=float(pinn_learning_rate)
                )

                collocation_t = torch.linspace(
                    0.0, float(t_end), int(pinn_collocation), device=device,
                ).reshape(-1, 1)
                initial_state = torch.tensor(
                    [[G0, P0]], dtype=torch.float32, device=device
                )

                plot_t = np.linspace(0.0, t_end, 300)
                G_reference = np.interp(plot_t, sol.t, sol.y[0])
                P_reference = np.interp(plot_t, sol.t, sol.y[1])

                status_placeholder = st.empty()
                progress_bar = st.progress(0.0)
                metric_cols = st.columns(4)
                total_metric = metric_cols[0].empty()
                physics_metric = metric_cols[1].empty()
                ic_metric = metric_cols[2].empty()
                mse_metric = metric_cols[3].empty()
                chart_placeholder = st.empty()

                epochs_seen, total_losses, ic_losses, physics_losses, mse_history = [], [], [], [], []

                for epoch in range(1, int(pinn_epochs) + 1):
                    optimizer.zero_grad()
                    total_loss, ic_loss, physics_loss = calculate_pinn_loss(
                        model, collocation_t, initial_state, p, n, m,
                        float(pinn_ic_weight), float(pinn_physics_weight),
                    )
                    total_loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                    optimizer.step()

                    should_update = (epoch == 1 or epoch % int(pinn_update_every) == 0 or epoch == int(pinn_epochs))
                    if should_update:
                        G_pred, P_pred = evaluate_pinn(model, plot_t, device)
                        prediction_mse = float(np.mean((G_pred - G_reference)**2) + np.mean((P_pred - P_reference)**2))
                        epochs_seen.append(epoch)
                        total_losses.append(float(total_loss.detach().cpu()))
                        ic_losses.append(float(ic_loss.detach().cpu()))
                        physics_losses.append(float(physics_loss.detach().cpu()))
                        mse_history.append(prediction_mse)

                        status_placeholder.write(f"Training on **{device.type.upper()}**: epoch **{epoch:,} / {int(pinn_epochs):,}**")
                        progress_bar.progress(epoch / int(pinn_epochs))
                        total_metric.metric("Total loss", f"{total_losses[-1]:.3e}")
                        physics_metric.metric("Physics loss", f"{physics_losses[-1]:.3e}")
                        ic_metric.metric("IC loss", f"{ic_losses[-1]:.3e}")
                        mse_improvement = 0.0 if mse_history[0] == 0 else 100.0 * (mse_history[0] - prediction_mse) / mse_history[0]
                        mse_metric.metric("ODE comparison MSE", f"{prediction_mse:.3e}", delta=f"{mse_improvement:.1f}% vs epoch 1")

                        live_figure = create_live_training_figure(
                            epochs_seen, total_losses, ic_losses, physics_losses,
                            plot_t, G_pred, P_pred, G_reference, P_reference,
                        )
                        chart_placeholder.pyplot(live_figure)
                        plt.close(live_figure)

                st.session_state["pinn_result"] = {
                    "t": plot_t, "G": G_pred, "P": P_pred,
                    "G_reference": G_reference, "P_reference": P_reference,
                    "epochs": epochs_seen, "total_losses": total_losses,
                    "ic_losses": ic_losses, "physics_losses": physics_losses,
                    "mse_history": mse_history, "device": device.type,
                }
                status_placeholder.success(
                    f"Training complete after {int(pinn_epochs):,} epochs on "
                    f"{device.type.upper()}. Final comparison MSE: {mse_history[-1]:.3e}"
                )

            elif "pinn_result" in st.session_state:
                result = st.session_state["pinn_result"]
                st.info("Showing the most recent trained PINN. Press **Train PINN** to retrain it with the current parameters.")
                saved_figure = create_live_training_figure(
                    result["epochs"], result["total_losses"], result["ic_losses"],
                    result["physics_losses"], result["t"], result["G"], result["P"],
                    result["G_reference"], result["P_reference"],
                )
                st.pyplot(saved_figure)
                plt.close(saved_figure)
                st.metric("Final ODE comparison MSE", f"{result['mse_history'][-1]:.3e}")

    # ── LNO Sub-Tab ──────────────────────────────────────────────────────
    with ml_tab_lno:
        st.subheader("Laplace Neural Operator")
        st.markdown(
            "A **data-driven** neural operator that maps initial conditions → full "
            "trajectories by learning in the Laplace/frequency domain. Unlike the PINN, "
            "the LNO trains on pre-generated BDF solutions and performs instant inference "
            "for any new initial condition."
        )

        with st.expander("LNO Hyperparameters", expanded=True):
            lc1, lc2, lc3 = st.columns(3)
            with lc1:
                lno_epochs = st.number_input("Epochs", min_value=1, max_value=5000, value=100, step=10, key="lno_epochs")
                lno_lr = st.number_input("Learning rate", min_value=1e-5, max_value=0.01, value=1e-3, format="%.5f", key="lno_lr")
                lno_samples = st.number_input("Training samples", min_value=50, max_value=5000, value=200, step=50, key="lno_samples")
            with lc2:
                lno_d_model = st.number_input("Latent width (d_model)", min_value=16, max_value=256, value=64, step=16, key="lno_d_model")
                lno_blocks = st.number_input("LNO blocks", min_value=1, max_value=12, value=4, step=1, key="lno_blocks")
                lno_modes = st.number_input("Fourier modes", min_value=8, max_value=256, value=32, step=8, key="lno_modes")
            with lc3:
                lno_timesteps = st.number_input("Time steps per trajectory", min_value=50, max_value=2000, value=201, step=50, key="lno_timesteps")
                lno_batch = st.number_input("Batch size", min_value=4, max_value=128, value=32, step=4, key="lno_batch")
                lno_seed = st.number_input("Random seed", min_value=0, max_value=100000, value=42, step=1, key="lno_seed")


        train_lno = st.button("Train LNO", type="primary", key="train_lno_btn")

        if train_lno:
            torch.manual_seed(int(lno_seed))
            np.random.seed(int(lno_seed))
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

            # Step 1: Generate training data
            data_status = st.empty()
            data_status.info(f"Generating {int(lno_samples)} BDF trajectories...")
            ics, trajectories = generate_training_data(
                p, n, m, t_end, int(lno_samples), int(lno_timesteps), seed=int(lno_seed),
            )
            data_status.success(f"Generated {int(lno_samples)} trajectories ({int(lno_timesteps)} steps each).")

            # Step 2: Prepare data
            train_loader, test_loader, norm_stats, t_norm = prepare_dataloaders(
                ics, trajectories, train_frac=0.8, batch_size=int(lno_batch), seed=int(lno_seed),
            )
            t_grid_dev = t_norm.to(device)

            # Step 3: Build model
            lno_model = LaplaceNeuralOperator(
                T=int(lno_timesteps), d_model=int(lno_d_model),
                n_lno_blocks=int(lno_blocks), n_modes=int(lno_modes),
                mlp_hidden=int(lno_d_model) * 2, lifting_layers=2, proj_layers=2,
                d_enc=max(32, int(lno_d_model) // 2),
            ).to(device)
            n_params = sum(pp.numel() for pp in lno_model.parameters() if pp.requires_grad)
            st.caption(f"Model: {n_params:,} trainable parameters")

            # Step 4: Train with live updates
            criterion = torch.nn.MSELoss()
            lno_optimizer = torch.optim.Adam(lno_model.parameters(), lr=float(lno_lr), weight_decay=1e-4)

            lno_status = st.empty()
            lno_progress = st.progress(0.0)
            lno_mcols = st.columns(3)
            lno_train_m = lno_mcols[0].empty()
            lno_test_m = lno_mcols[1].empty()
            lno_time_m = lno_mcols[2].empty()
            lno_chart = st.empty()

            ep_list, tr_list, te_list = [], [], []
            update_every = max(1, int(lno_epochs) // 20)

            for ep in range(1, int(lno_epochs) + 1):
                tr_loss = train_one_epoch(lno_model, train_loader, lno_optimizer, criterion, t_grid_dev, device)
                should_show = (ep == 1 or ep % update_every == 0 or ep == int(lno_epochs))
                if should_show:
                    te_loss = lno_evaluate(lno_model, test_loader, criterion, t_grid_dev, device)
                    ep_list.append(ep)
                    tr_list.append(tr_loss)
                    te_list.append(te_loss)

                    lno_status.write(f"Training on **{device.type.upper()}**: epoch **{ep:,} / {int(lno_epochs):,}**")
                    lno_progress.progress(ep / int(lno_epochs))
                    lno_train_m.metric("Train MSE", f"{tr_loss:.3e}")
                    lno_test_m.metric("Test MSE", f"{te_loss:.3e}")

                    # Predict on current ICs for live preview
                    t_pred, G_lno, P_lno, inf_ms = predict_sample(
                        lno_model, [G0, P0], t_norm, norm_stats, t_end, device,
                    )
                    lno_time_m.metric("Inference", f"{inf_ms:.1f} ms")

                    G_ref_lno = np.interp(t_pred, sol.t, sol.y[0]) if sol.success else np.zeros_like(t_pred)
                    P_ref_lno = np.interp(t_pred, sol.t, sol.y[1]) if sol.success else np.zeros_like(t_pred)

                    lno_fig = create_lno_training_figure(
                        ep_list, tr_list, te_list, t_pred, G_lno, P_lno, G_ref_lno, P_ref_lno,
                    )
                    lno_chart.pyplot(lno_fig)
                    plt.close(lno_fig)

            lno_status.success(f"LNO training complete — {int(lno_epochs):,} epochs. Final test MSE: {te_list[-1]:.3e}")

            st.session_state["lno_result"] = {
                "model_state": lno_model.state_dict(),
                "norm_stats": norm_stats, "t_norm": t_norm,
                "hp": {"T": int(lno_timesteps), "d_model": int(lno_d_model),
                       "n_lno_blocks": int(lno_blocks), "n_modes": int(lno_modes),
                       "mlp_hidden": int(lno_d_model)*2, "d_enc": max(32, int(lno_d_model)//2)},
                "epochs": ep_list, "train_losses": tr_list, "test_losses": te_list,
            }

        elif "lno_result" in st.session_state:
            res = st.session_state["lno_result"]
            st.info("Showing the most recent trained LNO. Press **Train LNO** to retrain.")
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            hp = res["hp"]
            lno_model = LaplaceNeuralOperator(
                T=hp["T"], d_model=hp["d_model"], n_lno_blocks=hp["n_lno_blocks"],
                n_modes=hp["n_modes"], mlp_hidden=hp["mlp_hidden"],
                lifting_layers=2, proj_layers=2, d_enc=hp["d_enc"],
            ).to(device)
            lno_model.load_state_dict(res["model_state"])

            t_pred, G_lno, P_lno, inf_ms = predict_sample(
                lno_model, [G0, P0], res["t_norm"], res["norm_stats"], t_end, device,
            )
            G_ref_lno = np.interp(t_pred, sol.t, sol.y[0]) if sol.success else np.zeros_like(t_pred)
            P_ref_lno = np.interp(t_pred, sol.t, sol.y[1]) if sol.success else np.zeros_like(t_pred)

            saved_lno_fig = create_lno_training_figure(
                res["epochs"], res["train_losses"], res["test_losses"],
                t_pred, G_lno, P_lno, G_ref_lno, P_ref_lno,
            )
            st.pyplot(saved_lno_fig)
            plt.close(saved_lno_fig)
            st.metric("Inference time", f"{inf_ms:.1f} ms")


# ═══════════════════════════════════════════════════════════════════════════
# Helper: shared relative L2 error computation
# ═══════════════════════════════════════════════════════════════════════════
def _relative_l2_error(y_pred, y_ref):
    """Compute per-step relative L2 error between predicted and reference.

    Parameters
    ----------
    y_pred : ndarray [2, T]  — solver output (row 0 = G, row 1 = P)
    y_ref  : ndarray [2, T]  — reference values at same time points

    Returns
    -------
    ndarray [T] — relative L2 error at each time step
    """
    ref_norms = np.sqrt(y_ref[0]**2 + y_ref[1]**2)
    ref_norms = np.where(ref_norms < 1e-15, 1.0, ref_norms)
    return np.sqrt((y_pred[0] - y_ref[0])**2 + (y_pred[1] - y_ref[1])**2) / ref_norms


with tab5:
    st.header("Benchmarking")
    bench_num, bench_ml = st.tabs(["Numerical Methods", "Machine Learning"])

    # ── Numerical Benchmarking ───────────────────────────────────────────
    with bench_num:
        st.subheader("Numerical Solver Comparison")
        st.markdown(
            "Compares all implemented ODE solvers against the SciPy LSODA reference "
            "using **relative L2 error**. Left panel sweeps step sizes; right panel "
            "shows per-step error at a single step size."
        )

        if not sol.success:
            st.error("Reference solution failed — benchmarking unavailable.")
        else:
            implemented = [s for s in ODE_SOLVERS if s.IS_IMPLEMENTED]
            if not implemented:
                st.info("No ODE solvers are implemented yet. See CONTRIBUTING.md to add one.")
            else:
                bcol1, bcol2 = st.columns(2)
                with bcol1:
                    bss_min = st.number_input("Min step size", min_value=0.001, value=0.01, format="%.4f", key="bss_min")
                    bss_max = st.number_input("Max step size", min_value=0.01, value=min(1.0, t_end / 2), format="%.4f", key="bss_max")
                    bss_count = st.number_input("Number of step sizes", min_value=2, max_value=30, value=8, step=1, key="bss_count")
                with bcol2:
                    bench_dt = st.number_input(
                        "Step size for per-step plot", min_value=0.001, max_value=float(t_end),
                        value=min(0.1, float(t_end) / 10), format="%.4f", key="bench_dt",
                    )

                run_bench = st.button("Run Benchmark", type="primary", key="run_bench_num")

                if run_bench:
                    colors = plt.cm.tab10.colors
                    dt_values = np.linspace(bss_min, bss_max, int(bss_count))

                    # ── Left: Step Size Sweep ────────────────────────────
                    fig_left, ax_left = plt.subplots(figsize=(7, 5))
                    # ── Right: Per-Step Error ────────────────────────────
                    fig_right, ax_right = plt.subplots(figsize=(7, 5))

                    summary_rows = []

                    for idx, solver_mod in enumerate(implemented):
                        c = colors[idx % len(colors)]

                        # Step-size sweep
                        max_errs_per_h = []
                        for dt_val in dt_values:
                            try:
                                t_s, y_s = solver_mod.solve(ode_system, t_span, [G0, P0], (p, n, m), float(dt_val))
                                G_i = np.interp(t_s, sol.t, sol.y[0])
                                P_i = np.interp(t_s, sol.t, sol.y[1])
                                errs = _relative_l2_error(y_s, np.array([G_i, P_i]))
                                max_errs_per_h.append(np.max(errs))
                            except Exception:
                                max_errs_per_h.append(np.nan)
                        max_errs_safe = np.maximum(np.array(max_errs_per_h, dtype=float), 1e-16)
                        ax_left.semilogy(dt_values, max_errs_safe, 'o-', color=c, linewidth=1.8, label=solver_mod.NAME)

                        # Per-step error at fixed dt (skip t=0 — all methods start exact)
                        try:
                            t_s2, y_s2 = solver_mod.solve(ode_system, t_span, [G0, P0], (p, n, m), float(bench_dt))
                            G_i2 = np.interp(t_s2, sol.t, sol.y[0])
                            P_i2 = np.interp(t_s2, sol.t, sol.y[1])
                            step_errs = _relative_l2_error(y_s2, np.array([G_i2, P_i2]))
                            step_errs_safe = np.maximum(step_errs[1:], 1e-16)
                            ax_right.semilogy(t_s2[1:], step_errs_safe, linewidth=1.8, color=c, label=solver_mod.NAME)
                            summary_rows.append({
                                "Method": solver_mod.NAME,
                                "Max Rel. Error": f"{np.max(step_errs[1:]):.3e}",
                                "Mean Rel. Error": f"{np.mean(step_errs[1:]):.3e}",
                            })
                        except Exception as exc:
                            st.warning(f"{solver_mod.NAME}: {exc}")

                    ax_left.set_xlabel("Step Size (h)")
                    ax_left.set_ylabel("Max Relative L2 Error (log)")
                    ax_left.set_title("Error vs Step Size — All Methods")
                    ax_left.legend(fontsize=8)
                    ax_left.grid(True, which="both", ls="--", alpha=0.5)

                    ax_right.set_xlabel("Time")
                    ax_right.set_ylabel("Relative L2 Error (log)")
                    ax_right.set_title(f"Per-Step Error — All Methods  (h = {bench_dt})")
                    ax_right.legend(fontsize=8)
                    ax_right.grid(True, which="both", ls="--", alpha=0.5)

                    col_l, col_r = st.columns(2)
                    with col_l:
                        st.pyplot(fig_left)
                        plt.close(fig_left)
                    with col_r:
                        st.pyplot(fig_right)
                        plt.close(fig_right)

                    if summary_rows:
                        st.markdown(f"#### Results at h = {bench_dt}")
                        st.dataframe(pd.DataFrame(summary_rows), use_container_width=True, hide_index=True)

    # ── ML Benchmarking ──────────────────────────────────────────────────
    with bench_ml:
        st.subheader("ML Model Comparison")
        st.markdown(
            "Compares trained ML models against the SciPy LSODA reference. "
            "If no model is trained yet, clicking **Run ML Benchmark** will train "
            "with the parameters set in the **Machine Learning** tab."
        )

        if not sol.success:
            st.error("Reference solution failed — benchmarking unavailable.")
        else:
            ml_bench_dt = st.number_input(
                "Evaluation time resolution", min_value=50, max_value=2000, value=300, step=50, key="ml_bench_res",
                help="Number of time points to evaluate the ML models at.",
            )

            run_ml_bench = st.button("Run ML Benchmark", type="primary", key="run_bench_ml")

            if run_ml_bench:
                device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
                ml_eval_t = np.linspace(0.0, t_end, int(ml_bench_dt))
                G_ref_ml = np.interp(ml_eval_t, sol.t, sol.y[0])
                P_ref_ml = np.interp(ml_eval_t, sol.t, sol.y[1])
                y_ref_ml = np.array([G_ref_ml, P_ref_ml])

                fig_ml_err, ax_ml_err = plt.subplots(figsize=(7, 5))
                ml_summary = []

                # ── PINN ─────────────────────────────────────────────
                if "pinn_result" not in st.session_state:
                    st.info("No trained PINN found — training one now with sidebar parameters...")
                    torch.manual_seed(42)
                    np.random.seed(42)
                    pinn_model = PINN(t_end=t_end, hidden_layers=3, neurons=32).to(device)
                    pinn_opt = torch.optim.Adam(pinn_model.parameters(), lr=0.01)
                    coll_t = torch.linspace(0.0, float(t_end), 200, device=device).reshape(-1, 1)
                    init_s = torch.tensor([[G0, P0]], dtype=torch.float32, device=device)
                    pinn_bar = st.progress(0.0, text="Training PINN...")
                    for ep in range(1, 1001):
                        pinn_opt.zero_grad()
                        tl, il, pl = calculate_pinn_loss(pinn_model, coll_t, init_s, p, n, m, 10.0, 1.0)
                        tl.backward()
                        torch.nn.utils.clip_grad_norm_(pinn_model.parameters(), max_norm=10.0)
                        pinn_opt.step()
                        if ep % 100 == 0:
                            pinn_bar.progress(ep / 1000, text=f"Training PINN... epoch {ep}/1000")
                    pinn_bar.empty()
                    G_pinn, P_pinn = evaluate_pinn(pinn_model, ml_eval_t, device)
                    st.success("PINN auto-trained (1000 epochs).")
                else:
                    pr = st.session_state["pinn_result"]
                    G_pinn = np.interp(ml_eval_t, pr["t"], pr["G"])
                    P_pinn = np.interp(ml_eval_t, pr["t"], pr["P"])

                y_pinn = np.array([G_pinn, P_pinn])
                pinn_errs = _relative_l2_error(y_pinn, y_ref_ml)
                pinn_errs_safe = np.maximum(pinn_errs, 1e-16)
                ax_ml_err.semilogy(ml_eval_t, pinn_errs_safe, linewidth=2, label="PINN", color="darkorange")
                ml_summary.append({"Model": "PINN", "Max Rel. Error": f"{np.max(pinn_errs):.3e}", "Mean Rel. Error": f"{np.mean(pinn_errs):.3e}"})

                # ── LNO ──────────────────────────────────────────────
                if "lno_result" not in st.session_state:
                    st.info("No trained LNO found — training one now with sidebar parameters...")
                    torch.manual_seed(42)
                    np.random.seed(42)
                    lno_n_ts = 201
                    ics_d, traj_d = generate_training_data(p, n, m, t_end, 200, lno_n_ts, seed=42)
                    tr_ld, te_ld, ns, t_n = prepare_dataloaders(ics_d, traj_d, batch_size=32, seed=42)
                    t_gd = t_n.to(device)
                    lno_auto = LaplaceNeuralOperator(
                        T=lno_n_ts, d_model=64, n_lno_blocks=4, n_modes=32,
                        mlp_hidden=128, lifting_layers=2, proj_layers=2, d_enc=32,
                    ).to(device)
                    lno_opt = torch.optim.Adam(lno_auto.parameters(), lr=1e-3, weight_decay=1e-4)
                    crit = torch.nn.MSELoss()
                    lno_bar = st.progress(0.0, text="Training LNO...")
                    for ep in range(1, 101):
                        train_one_epoch(lno_auto, tr_ld, lno_opt, crit, t_gd, device)
                        if ep % 10 == 0:
                            lno_bar.progress(ep / 100, text=f"Training LNO... epoch {ep}/100")
                    lno_bar.empty()
                    st.success("LNO auto-trained (100 epochs on 200 samples).")
                    t_lno_b, G_lno_b, P_lno_b, _ = predict_sample(lno_auto, [G0, P0], t_n, ns, t_end, device)
                else:
                    res = st.session_state["lno_result"]
                    hp = res["hp"]
                    lno_auto = LaplaceNeuralOperator(
                        T=hp["T"], d_model=hp["d_model"], n_lno_blocks=hp["n_lno_blocks"],
                        n_modes=hp["n_modes"], mlp_hidden=hp["mlp_hidden"],
                        lifting_layers=2, proj_layers=2, d_enc=hp["d_enc"],
                    ).to(device)
                    lno_auto.load_state_dict(res["model_state"])
                    t_lno_b, G_lno_b, P_lno_b, _ = predict_sample(lno_auto, [G0, P0], res["t_norm"], res["norm_stats"], t_end, device)

                G_lno_interp = np.interp(ml_eval_t, t_lno_b, G_lno_b)
                P_lno_interp = np.interp(ml_eval_t, t_lno_b, P_lno_b)
                y_lno = np.array([G_lno_interp, P_lno_interp])
                lno_errs = _relative_l2_error(y_lno, y_ref_ml)
                lno_errs_safe = np.maximum(lno_errs, 1e-16)
                ax_ml_err.semilogy(ml_eval_t, lno_errs_safe, linewidth=2, label="LNO", color="mediumseagreen")
                ml_summary.append({"Model": "LNO", "Max Rel. Error": f"{np.max(lno_errs):.3e}", "Mean Rel. Error": f"{np.mean(lno_errs):.3e}"})

                ax_ml_err.set_xlabel("Time")
                ax_ml_err.set_ylabel("Relative L2 Error (log)")
                ax_ml_err.set_title("ML Per-Step Error — PINN vs LNO")
                ax_ml_err.legend(fontsize=9)
                ax_ml_err.grid(True, which="both", ls="--", alpha=0.5)

                st.pyplot(fig_ml_err)
                plt.close(fig_ml_err)

                st.markdown("#### Results")
                st.dataframe(pd.DataFrame(ml_summary), use_container_width=True, hide_index=True)

