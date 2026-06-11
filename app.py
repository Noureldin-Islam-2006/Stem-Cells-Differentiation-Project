import streamlit as st
import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp
import matplotlib.patches as mpatches

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
    st.header("Machine Learning (Placeholder)")
    st.info("Physics-Informed Neural Network (PINN) implementation coming soon.")
