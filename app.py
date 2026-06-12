import warnings

import streamlit as st
import numpy as np
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
tab1, tab2, tab3, tab4 = st.tabs(["System Dynamics", "Phase Plane & Steady States", "Convergence Analysis", "Machine Learning"])

with tab1:
    st.header("System Dynamics")

    # Build sub-tab names from the solver registry
    method_names = ["Reference (SciPy)"] + [s.NAME for s in ODE_SOLVERS]
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
        else:
            st.error("ODE Solver failed to converge.")

    # --- ODE method sub-tabs (driven by registry) ---
    for idx, solver_mod in enumerate(ODE_SOLVERS):
        with method_tabs[idx + 1]:
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
    st.success(f"**Found Steady State (Root):** G = {root[0]:.4f}, P = {root[1]:.4f}")

with tab3:
    st.header("Convergence Analysis")

    # Build method list: Newton + implemented ODE solvers
    conv_methods = ["Newton-Raphson"] + [s.NAME for s in ODE_SOLVERS if s.IS_IMPLEMENTED]
    selected_conv = st.selectbox("Method to analyse", conv_methods, key="conv_method")

    if selected_conv == "Newton-Raphson":
        st.markdown("Residual L2 norm at each Newton-Raphson iteration (linear scale).")
        nr_tol = st.number_input("Tolerance", min_value=1e-12, max_value=1.0, value=1e-6, format="%.1e", key="nr_tol")
        nr_max = st.number_input("Max iterations", min_value=1, max_value=500, value=50, step=1, key="nr_max")
        with warnings.catch_warnings(record=True) as cw:
            warnings.simplefilter("always")
            _, _, nr_errors = solve_newton([G0, P0], p, n, m, tolerance=nr_tol, max_iter=nr_max)
        for w in cw:
            st.warning(str(w.message))
        if len(nr_errors) > 0:
            fig5, ax5 = plt.subplots(figsize=(8, 5))
            ax5.plot(range(1, len(nr_errors) + 1), nr_errors, 'mo-', linewidth=2)
            ax5.set_xlabel('Iteration')
            ax5.set_ylabel('L2 Norm of Residual')
            ax5.set_title('Newton-Raphson Iteration Error')
            ax5.grid(True, ls="--")
            st.pyplot(fig5)
        else:
            st.write("Initial guess is already a root — no iterations needed.")

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
                        err = np.max(np.sqrt((y_s[0] - G_interp)**2 + (y_s[1] - P_interp)**2))
                        ss_errors.append(err)
                    except Exception:
                        ss_errors.append(np.nan)

                fig_ss, ax_ss = plt.subplots(figsize=(6, 4))
                ax_ss.plot(dt_values, ss_errors, 'rs-', linewidth=2)
                ax_ss.set_xlabel('Step Size (h)')
                ax_ss.set_ylabel('Max L2 Error vs Reference')
                ax_ss.set_title('Step Size Error')
                ax_ss.grid(True, ls="--")
                st.pyplot(fig_ss)

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
                    iter_errs = np.sqrt((y_it[0] - G_ref_it)**2 + (y_it[1] - P_ref_it)**2)

                    fig_it, ax_it = plt.subplots(figsize=(6, 4))
                    ax_it.plot(t_it, iter_errs, 'b-', linewidth=2)
                    ax_it.set_xlabel('Time')
                    ax_it.set_ylabel('L2 Error vs Reference')
                    ax_it.set_title(f'Per-Step Error  (dt = {iter_dt})')
                    ax_it.grid(True, ls="--")
                    st.pyplot(fig_it)
                except Exception as exc:
                    st.error(f"Solver error: {exc}")

with tab4:
    st.header("Physics-Informed Neural Network")
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
                0.0,
                float(t_end),
                int(pinn_collocation),
                device=device,
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

            epochs_seen = []
            total_losses = []
            ic_losses = []
            physics_losses = []
            mse_history = []

            for epoch in range(1, int(pinn_epochs) + 1):
                optimizer.zero_grad()
                total_loss, ic_loss, physics_loss = calculate_pinn_loss(
                    model,
                    collocation_t,
                    initial_state,
                    p,
                    n,
                    m,
                    float(pinn_ic_weight),
                    float(pinn_physics_weight),
                )
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
                optimizer.step()

                should_update = (
                    epoch == 1
                    or epoch % int(pinn_update_every) == 0
                    or epoch == int(pinn_epochs)
                )
                if should_update:
                    G_pred, P_pred = evaluate_pinn(model, plot_t, device)
                    prediction_mse = float(
                        np.mean((G_pred - G_reference) ** 2)
                        + np.mean((P_pred - P_reference) ** 2)
                    )

                    epochs_seen.append(epoch)
                    total_losses.append(float(total_loss.detach().cpu()))
                    ic_losses.append(float(ic_loss.detach().cpu()))
                    physics_losses.append(float(physics_loss.detach().cpu()))
                    mse_history.append(prediction_mse)

                    status_placeholder.write(
                        f"Training on **{device.type.upper()}**: epoch "
                        f"**{epoch:,} / {int(pinn_epochs):,}**"
                    )
                    progress_bar.progress(epoch / int(pinn_epochs))
                    total_metric.metric("Total loss", f"{total_losses[-1]:.3e}")
                    physics_metric.metric(
                        "Physics loss", f"{physics_losses[-1]:.3e}"
                    )
                    ic_metric.metric("IC loss", f"{ic_losses[-1]:.3e}")
                    mse_improvement = (
                        0.0
                        if mse_history[0] == 0
                        else 100.0 * (mse_history[0] - prediction_mse) / mse_history[0]
                    )
                    mse_metric.metric(
                        "ODE comparison MSE",
                        f"{prediction_mse:.3e}",
                        delta=f"{mse_improvement:.1f}% vs epoch 1",
                    )

                    live_figure = create_live_training_figure(
                        epochs_seen,
                        total_losses,
                        ic_losses,
                        physics_losses,
                        plot_t,
                        G_pred,
                        P_pred,
                        G_reference,
                        P_reference,
                    )
                    chart_placeholder.pyplot(live_figure)
                    plt.close(live_figure)

            st.session_state["pinn_result"] = {
                "t": plot_t,
                "G": G_pred,
                "P": P_pred,
                "G_reference": G_reference,
                "P_reference": P_reference,
                "epochs": epochs_seen,
                "total_losses": total_losses,
                "ic_losses": ic_losses,
                "physics_losses": physics_losses,
                "mse_history": mse_history,
                "device": device.type,
            }
            status_placeholder.success(
                f"Training complete after {int(pinn_epochs):,} epochs on "
                f"{device.type.upper()}. Final comparison MSE: {mse_history[-1]:.3e}"
            )

        elif "pinn_result" in st.session_state:
            result = st.session_state["pinn_result"]
            st.info(
                "Showing the most recent trained PINN. Press **Train PINN** to "
                "retrain it with the current parameters."
            )
            saved_figure = create_live_training_figure(
                result["epochs"],
                result["total_losses"],
                result["ic_losses"],
                result["physics_losses"],
                result["t"],
                result["G"],
                result["P"],
                result["G_reference"],
                result["P_reference"],
            )
            st.pyplot(saved_figure)
            plt.close(saved_figure)
            st.metric("Final ODE comparison MSE", f"{result['mse_history'][-1]:.3e}")
