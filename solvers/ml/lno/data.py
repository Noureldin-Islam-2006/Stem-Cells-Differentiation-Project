"""
Data generation and preprocessing for LNO training.

Generates BDF trajectories by sampling random initial conditions and solving
the gene regulatory network ODE, then normalises for training.
"""

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader, random_split
from scipy.integrate import solve_ivp

from solvers.core import ode_system


def generate_training_data(p, n, m, t_end, n_samples, n_timesteps, seed=42):
    """Generate BDF trajectories from random initial conditions.

    Parameters
    ----------
    p : dict — model parameters
    n, m : float — Hill coefficients
    t_end : float — end time
    n_samples : int — number of IC samples
    n_timesteps : int — time resolution per trajectory
    seed : int

    Returns
    -------
    ics : ndarray [n_samples, 2]
    trajectories : ndarray [n_samples, n_timesteps, 2]
    """
    rng = np.random.RandomState(seed)
    t_eval = np.linspace(0, t_end, n_timesteps)

    ics = np.zeros((n_samples, 2), dtype=np.float32)
    trajectories = np.zeros((n_samples, n_timesteps, 2), dtype=np.float32)

    # Sample ICs from a reasonable range around the system's steady states
    for i in range(n_samples):
        G0 = rng.uniform(0.01, 5.0)
        P0 = rng.uniform(0.01, 5.0)
        ics[i] = [G0, P0]

        sol = solve_ivp(
            ode_system, (0, t_end), [G0, P0],
            args=(p, n, m), t_eval=t_eval, method='BDF',
            rtol=1e-6, atol=1e-8,
        )
        if sol.success:
            trajectories[i, :, 0] = sol.y[0]
            trajectories[i, :, 1] = sol.y[1]
        else:
            # Fallback to LSODA if BDF fails
            sol = solve_ivp(
                ode_system, (0, t_end), [G0, P0],
                args=(p, n, m), t_eval=t_eval, method='LSODA',
            )
            trajectories[i, :, 0] = sol.y[0]
            trajectories[i, :, 1] = sol.y[1]

    return ics, trajectories


def prepare_dataloaders(
    ics, trajectories, train_frac=0.8, batch_size=32, seed=42,
):
    """Normalise data and create train/test DataLoaders.

    Returns
    -------
    train_loader, test_loader : DataLoader
    norm_stats : dict with ic_mean, ic_std
    t_norm : Tensor [T] — normalised time grid [0, 1]
    """
    N, T, S = trajectories.shape

    # Normalise ICs (per-species mean/std)
    ic_mean = ics.mean(axis=0)
    ic_std = ics.std(axis=0) + 1e-8
    ics_norm = (ics - ic_mean) / ic_std

    # Normalise trajectories with same stats
    traj_norm = (trajectories - ic_mean[None, None, :]) / ic_std[None, None, :]

    norm_stats = dict(ic_mean=ic_mean, ic_std=ic_std)
    t_norm = torch.linspace(0.0, 1.0, T, dtype=torch.float32)

    X = torch.from_numpy(ics_norm)
    Y = torch.from_numpy(traj_norm)

    n_train = int(train_frac * N)
    n_test = N - n_train

    dataset = TensorDataset(X, Y)
    train_ds, test_ds = random_split(
        dataset, [n_train, n_test],
        generator=torch.Generator().manual_seed(seed),
    )

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    return train_loader, test_loader, norm_stats, t_norm
