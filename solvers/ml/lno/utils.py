"""
Visualization helpers for the Laplace Neural Operator.
"""

import numpy as np
import matplotlib.pyplot as plt


def create_lno_training_figure(
    epochs_seen, train_losses, test_losses,
    t_values, G_pred, P_pred, G_reference, P_reference,
):
    """Build a two-panel figure: loss curves + prediction vs reference.

    Returns
    -------
    matplotlib.figure.Figure
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))

    axes[0].semilogy(epochs_seen, train_losses, label="Train MSE", linewidth=2, color="royalblue")
    axes[0].semilogy(epochs_seen, test_losses, label="Test MSE", linewidth=2, color="tomato")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("MSE Loss (log scale)")
    axes[0].set_title("LNO Learning Progress")
    axes[0].grid(True, which="both", linestyle="--", alpha=0.5)
    axes[0].legend()

    axes[1].plot(t_values, G_reference, "r--", label="G reference (BDF)", alpha=0.75)
    axes[1].plot(t_values, P_reference, "b--", label="P reference (BDF)", alpha=0.75)
    axes[1].plot(t_values, G_pred, color="darkred", label="G LNO", linewidth=2)
    axes[1].plot(t_values, P_pred, color="darkblue", label="P LNO", linewidth=2)
    axes[1].set_xlabel("Time")
    axes[1].set_ylabel("Concentration")
    axes[1].set_title(f"LNO Prediction at Epoch {epochs_seen[-1]}")
    axes[1].grid(True, alpha=0.4)
    axes[1].legend()

    fig.tight_layout()
    return fig
