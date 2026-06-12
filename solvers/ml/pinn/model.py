"""
PINN (Physics-Informed Neural Network) architecture for the gene regulatory network.
"""

import torch.nn as nn
import torch.nn.functional as F


class PINN(nn.Module):
    """Neural network that learns the positive concentrations G(t) and P(t).

    The network maps a scalar time input to two output concentrations,
    enforcing positivity via a softplus activation on the output layer.
    Time is normalized to [-1, 1] before being fed into the network.

    Parameters
    ----------
    t_end : float
        End of the time domain (used for input normalization).
    hidden_layers : int
        Number of hidden layers.
    neurons : int
        Number of neurons per hidden layer.
    """

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
