"""
Training and evaluation functions for the Laplace Neural Operator.
"""

import time

import numpy as np
import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau


def train_one_epoch(model, train_loader, optimizer, criterion, t_grid, device):
    """Run one training epoch. Returns average loss."""
    model.train()
    running_loss = 0.0
    for ic_batch, traj_batch in train_loader:
        ic_batch = ic_batch.to(device)
        traj_batch = traj_batch.to(device)

        optimizer.zero_grad()
        pred = model(ic_batch, t_grid)
        loss = criterion(pred, traj_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * ic_batch.size(0)

    return running_loss / len(train_loader.dataset)


def evaluate(model, test_loader, criterion, t_grid, device):
    """Evaluate on test set. Returns average loss."""
    model.eval()
    running_loss = 0.0
    with torch.no_grad():
        for ic_batch, traj_batch in test_loader:
            ic_batch = ic_batch.to(device)
            traj_batch = traj_batch.to(device)
            pred = model(ic_batch, t_grid)
            loss = criterion(pred, traj_batch)
            running_loss += loss.item() * ic_batch.size(0)
    return running_loss / len(test_loader.dataset)


def predict_sample(model, ic, t_grid, norm_stats, t_end, device):
    """Run LNO inference on a single initial condition.

    Parameters
    ----------
    model : LaplaceNeuralOperator
    ic : array-like [2] — raw (un-normalised) initial condition [G0, P0]
    t_grid : Tensor [T] — normalised time grid
    norm_stats : dict — ic_mean, ic_std
    t_end : float
    device : torch.device

    Returns
    -------
    t_arr : ndarray [T]
    G_pred, P_pred : ndarray [T] — de-normalised predictions
    inference_ms : float — inference time in milliseconds
    """
    model.eval()
    ic_mean = norm_stats["ic_mean"]
    ic_std = norm_stats["ic_std"]

    ic_norm = (np.array(ic, dtype=np.float32) - ic_mean) / ic_std
    ic_tensor = torch.from_numpy(ic_norm).unsqueeze(0).to(device)
    t_grid = t_grid.to(device)

    t0 = time.perf_counter()
    with torch.no_grad():
        pred = model(ic_tensor, t_grid)
    inference_ms = (time.perf_counter() - t0) * 1000.0

    pred_np = pred[0].cpu().numpy()
    pred_phys = pred_np * ic_std[None, :] + ic_mean[None, :]

    T = t_grid.shape[0]
    t_arr = np.linspace(0, t_end, T)

    return t_arr, pred_phys[:, 0], pred_phys[:, 1], inference_ms
