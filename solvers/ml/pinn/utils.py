"""
Utility functions for PINN evaluation and visualization.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch


def evaluate_pinn(model, t_values, device):
    """Run the trained PINN in inference mode over the given time values.

    Parameters
    ----------
    model : nn.Module
        Trained PINN model.
    t_values : numpy.ndarray
        1-D array of time points.
    device : torch.device
        Device the model lives on.

    Returns
    -------
    G_pred : numpy.ndarray
    P_pred : numpy.ndarray
    """
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
    """Build a two-panel matplotlib figure showing training progress.

    Left panel: loss curves (log scale).
    Right panel: PINN predictions vs. ODE reference.

    Returns
    -------
    matplotlib.figure.Figure
    """
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
