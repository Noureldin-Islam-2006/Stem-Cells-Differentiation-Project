"""
Physics-informed loss functions for PINN training.

Provides the Torch-compatible ODE right-hand side and the combined
initial-condition + physics loss used during training.
"""

import torch


def torch_ode_rhs(G, P, p, n, m):
    """Torch version of the governing equations used in the PINN loss.

    Parameters
    ----------
    G, P : torch.Tensor
        Current concentrations (with grad tracking).
    p : dict
        Model parameters.
    n, m : float
        Hill coefficients.

    Returns
    -------
    tuple of torch.Tensor
        (dG/dt, dP/dt)
    """
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
    """Compute the combined PINN training loss.

    Parameters
    ----------
    model : nn.Module
        The PINN model.
    collocation_t : torch.Tensor
        Time points for physics residual evaluation.
    initial_state : torch.Tensor
        Target initial concentrations [[G0, P0]].
    p : dict
        Model parameters.
    n, m : float
        Hill coefficients.
    ic_weight : float
        Weight for the initial-condition loss term.
    physics_weight : float
        Weight for the physics (ODE residual) loss term.

    Returns
    -------
    total_loss : torch.Tensor
    ic_loss : torch.Tensor
    physics_loss : torch.Tensor
    """
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
