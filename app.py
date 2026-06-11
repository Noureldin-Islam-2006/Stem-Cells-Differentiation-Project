import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import torch
import torch.nn as nn
import torch.nn.functional as F

# --- Core Mathematical Logic ---

def calculate_terms(G, P, p, n, m):
    Gterm1 = p['a1'] * (G**n) / (p['tha1']**n + G**n)
    Gterm2 = p['b1'] * (p['thb1']**m) / (p['thb1']**m + (G**m) * (P**m))
    Gterm3 = -p['k1'] * G
    Pterm1 = p['a2'] * (P**n) / (p['tha2']**n + P**n)
    Pterm2 = p['b2'] * (p['thb2']**m) / (p['thb2']**m + (G**m) * (P**m))
    Pterm3 = -p['k2'] * P
    return Gterm1, Gterm2, Gterm3, Pterm1, Pterm2, Pterm3

def ode_system(t, vars, p, n, m):
    G, P = vars
    g1, g2, g3, p1, p2, p3 = calculate_terms(G, P, p, n, m)
    return [g1 + g2 + g3, p1 + p2 + p3]

def algebraic_system(vars, p, n, m):
    return np.array(ode_system(0, vars, p, n, m))

def compute_jacobian(func, vars, epsilon=1e-5):
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
    v = np.array(initial_guess, dtype=float)
    history = [v.copy()]
    errors = []
    for i in range(max_iter):
        F_val = algebraic_system(v, p, n, m)
        err_norm = np.linalg.norm(F_val)
        errors.append(err_norm)
        if err_norm < tolerance: break
        J = compute_jacobian(lambda x: algebraic_system(x, p, n, m), v)
        # Add small damping or pseudo-inverse if singular, though standard inverse is usually fine
        try:
            delta = np.linalg.solve(J, -F_val)
        except np.linalg.LinAlgError:
            st.warning("Jacobian is singular, using pseudo-inverse.")
            delta = np.linalg.pinv(J).dot(-F_val)
        v = v + delta
        history.append(v.copy())
    return v, np.array(history), errors


# --- Physics-Informed Neural Network Logic ---

class PINN(nn.Module):
    """Neural network that learns the positive concentrations G(t) and P(t)."""

    def __init__(self, t_end, hidden_layers=3, neurons=32):
        super().__init__()
        self.t_end = max(float(t_end), 1e-6)

        layers = [nn.Linear(1, neurons), nn.Tanh()]
        for _ in range(hidden_layers - 1):
            layers.extend([nn.Linear(neurons, neurons), nn.Tanh()])
        layers.append(nn.Linear(neurons, 2))
        self.network = nn.Sequential(*layers)

    def forward(self, t):
        # Normalization keeps the network input in a training-friendly range.
        normalized_t = 2.0 * t / self.t_end - 1.0
        return F.softplus(self.network(normalized_t)) + 1e-6


def torch_ode_rhs(G, P, p, n, m):
    """Torch version of the governing equations used in the PINN loss."""
    g_activation = p['a1'] * G.pow(n) / (p['tha1']**n + G.pow(n))
    g_inhibition = (
        p['b1'] * p['thb1']**m
        / (p['thb1']**m + G.pow(m) * P.pow(m))
    )
    p_activation = p['a2'] * P.pow(n) / (p['tha2']**n + P.pow(n))
    p_inhibition = (
        p['b2'] * p['thb2']**m
        / (p['thb2']**m + G.pow(m) * P.pow(m))
    )
    return (
        g_activation + g_inhibition - p['k1'] * G,
        p_activation + p_inhibition - p['k2'] * P,
    )


def calculate_pinn_loss(
    model,
    collocation_t,
    initial_state,
    p,
    n,
    m,
    ic_weight,
    physics_weight,
):
    collocation_t.requires_grad_(True)
    prediction = model(collocation_t)
    G = prediction[:, 0:1]
    P = prediction[:, 1:2]

    dG_dt = torch.autograd.grad(
        G,
        collocation_t,
        grad_outputs=torch.ones_like(G),
        create_graph=True,
        retain_graph=True,
    )[0]
    dP_dt = torch.autograd.grad(
        P,
        collocation_t,
        grad_outputs=torch.ones_like(P),
        create_graph=True,
    )[0]

    rhs_G, rhs_P = torch_ode_rhs(G, P, p, n, m)
    physics_loss = torch.mean((dG_dt - rhs_G) ** 2) + torch.mean(
        (dP_dt - rhs_P) ** 2
    )

    t0 = torch.zeros((1, 1), dtype=torch.float32, device=collocation_t.device)
    initial_prediction = model(t0)
    ic_loss = torch.mean((initial_prediction - initial_state) ** 2)
    total_loss = ic_weight * ic_loss + physics_weight * physics_loss
    return total_loss, ic_loss, physics_loss


def evaluate_pinn(model, t_values, device):
    model.eval()
    with torch.no_grad():
        t_tensor = torch.tensor(
            t_values.reshape(-1, 1), dtype=torch.float32, device=device
        )
        prediction = model(t_tensor).cpu().numpy()
    model.train()
    return prediction[:, 0], prediction[:, 1]


def create_live_training_figure(
    epochs_seen,
    total_losses,
    ic_losses,
    physics_losses,
    t_values,
    G_pred,
    P_pred,
    G_reference,
    P_reference,
):
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].semilogy(epochs_seen, total_losses, label="Total loss", linewidth=2)
    axes[0].semilogy(epochs_seen, ic_losses, label="Initial condition loss")
    axes[0].semilogy(epochs_seen, physics_losses, label="Physics loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss (log scale)")
    axes[0].set_title("PINN Learning Progress")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(t_values, G_reference, "r--", label="G ODE reference", alpha=0.75)
    axes[1].plot(t_values, P_reference, "b--", label="P ODE reference", alpha=0.75)
    axes[1].plot(t_values, G_pred, color="darkred", label="G PINN", linewidth=2)
    axes[1].plot(t_values, P_pred, color="darkblue", label="P PINN", linewidth=2)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Concentration")
    axes[1].set_title(f"Prediction at Epoch {epochs_seen[-1]}")
    axes[1].grid(True, alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    return fig

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

# 2. Main Panel (Tabs)
tab1, tab2, tab3, tab4 = st.tabs(["System Dynamics", "Phase Plane & Steady States", "Convergence Analysis", "Machine Learning"])

with tab1:
    st.header("System Dynamics (ODE Solver)")
    
    t_span = (0, t_end)
    t_eval = np.linspace(0, t_end, max(1000, int(t_end*10)))
    sol = solve_ivp(ode_system, t_span, [G0, P0], args=(p, n, m), t_eval=t_eval, method='LSODA')
    
    if sol.success:
        fig1, ax1 = plt.subplots(figsize=(10, 5))
        ax1.plot(sol.t, sol.y[0], label='GATA-1 (G)', color='red', linewidth=2)
        ax1.plot(sol.t, sol.y[1], label='PU.1 (P)', color='blue', linewidth=2)
        ax1.set_xlabel('Time')
        ax1.set_ylabel('Concentration')
        ax1.set_title('Protein Concentrations over Time')
        ax1.legend()
        ax1.grid(True)
        st.pyplot(fig1)
        
        # Breakdown into component terms
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

with tab2:
    st.header("Phase Plane & Steady States (Newton-Raphson)")
    
    root, history, errors = solve_newton([G0, P0], p, n, m)
    
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
    st.markdown("Quadratic convergence visualization using the L2 norm of the Newton-Raphson residual error over iterations on a logarithmic scale.")
    
    if len(errors) > 0:
        fig5, ax5 = plt.subplots(figsize=(8, 5))
        ax5.plot(range(1, len(errors) + 1), errors, 'mo-', linewidth=2)
        ax5.set_yscale('log')
        ax5.set_xlabel('Iteration')
        ax5.set_ylabel('L2 Norm of Error (Log Scale)')
        ax5.set_title('Newton-Raphson Error Convergence')
        ax5.grid(True, which="both", ls="--")
        st.pyplot(fig5)
    else:
        st.write("No iterations were performed (perhaps the initial guess was already a root).")

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
