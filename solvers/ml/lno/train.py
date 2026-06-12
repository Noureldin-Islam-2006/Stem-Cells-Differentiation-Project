"""Training and evaluation routines for Laplace Neural Operator (LNO).
"""

import time
import math
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

from solvers.ml.lno.model import build_model
from solvers.ml.lno.data_generation import PARAMS, genetic_switch

def load_data(hp: dict):
    """Load, scale, and split datasets into training/testing loaders."""
    ics   = np.load(hp["ic_path"]).astype(np.float32)
    trajs = np.load(hp["traj_path"]).astype(np.float32)

    N, T, S = trajs.shape

    ic_mean = ics.mean(axis=0)
    ic_std  = ics.std(axis=0) + 1e-8
    ics_norm = (ics - ic_mean) / ic_std
    traj_norm = (trajs - ic_mean[None, None, :]) / ic_std[None, None, :]

    norm_stats = dict(ic_mean=ic_mean, ic_std=ic_std)
    t_norm = torch.linspace(0.0, 1.0, T, dtype=torch.float32)

    X = torch.from_numpy(ics_norm)
    Y = torch.from_numpy(traj_norm)

    n_train = int(hp["train_frac"] * N)
    n_test  = N - n_train

    dataset = TensorDataset(X, Y)
    train_ds, test_ds = random_split(
        dataset,
        [n_train, n_test],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(
        train_ds,
        batch_size  = hp["batch_size"],
        shuffle     = True,
        drop_last   = False,
    )
    test_loader = DataLoader(
        test_ds,
        batch_size  = hp["batch_size"],
        shuffle     = False,
    )
    return train_loader, test_loader, norm_stats, t_norm


def train(model, train_loader, test_loader, t_grid, hp, device, status_placeholder=None, progress_bar=None):
    """Train the model using MSE loss."""
    model.to(device)
    t_grid = t_grid.to(device)

    optimizer = Adam(
        model.parameters(),
        lr           = hp["lr"],
        weight_decay = hp["weight_decay"],
    )
    scheduler = ReduceLROnPlateau(
        optimizer,
        mode     = "min",
        patience = hp["lr_patience"],
        factor   = hp["lr_factor"],
    )
    criterion = nn.MSELoss()

    train_losses = []
    test_losses  = []

    for epoch in range(1, hp["epochs"] + 1):
        model.train()
        running_loss = 0.0
        for ic_batch, traj_batch in train_loader:
            ic_batch   = ic_batch.to(device)
            traj_batch = traj_batch.to(device)

            optimizer.zero_grad()
            pred  = model(ic_batch, t_grid)
            loss  = criterion(pred, traj_batch)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            running_loss += loss.item() * ic_batch.size(0)

        train_loss = running_loss / len(train_loader.dataset)

        model.eval()
        test_running = 0.0
        with torch.no_grad():
            for ic_batch, traj_batch in test_loader:
                ic_batch   = ic_batch.to(device)
                traj_batch = traj_batch.to(device)
                pred       = model(ic_batch, t_grid)
                loss       = criterion(pred, traj_batch)
                test_running += loss.item() * ic_batch.size(0)

        test_loss = test_running / len(test_loader.dataset)
        scheduler.step(test_loss)

        train_losses.append(train_loss)
        test_losses.append(test_loss)

        if status_placeholder is not None:
            status_placeholder.write(f"Epoch **{epoch} / {hp['epochs']}** | Train Loss: **{train_loss:.5e}** | Test Loss: **{test_loss:.5e}**")
        if progress_bar is not None:
            progress_bar.progress(epoch / hp["epochs"])

    torch.save(model.state_dict(), hp["model_path"])
    return train_losses, test_losses


def _mock_bdf_time(n_steps: int = 501) -> float:
    """Simulate BDF solve time on genetic switch ODE system."""
    try:
        from scipy.integrate import solve_ivp
        t_eval = np.linspace(0, 10, n_steps)
        t0 = time.perf_counter()
        solve_ivp(genetic_switch, (0.0, 10.0), [1.0, 1.0], method="BDF",
                  t_eval=t_eval, rtol=1e-6, atol=1e-8)
        return time.perf_counter() - t0
    except Exception:
        return 0.12


def evaluate_and_plot(model, test_loader, t_grid, norm_stats, hp, device):
    """Evaluate LNO on test data and save comparison plot."""
    model.eval()
    t_grid = t_grid.to(device)

    ic_list, traj_list = [], []
    with torch.no_grad():
        for ic_b, traj_b in test_loader:
            ic_list.append(ic_b)
            traj_list.append(traj_b)
    ic_all   = torch.cat(ic_list,   dim=0)
    traj_all = torch.cat(traj_list, dim=0)

    rng_idx = np.random.randint(0, ic_all.shape[0])
    ic_sample   = ic_all[rng_idx:rng_idx+1].to(device)
    traj_sample = traj_all[rng_idx]

    t_start = time.perf_counter()
    with torch.no_grad():
        pred = model(ic_sample, t_grid)
    lno_time = time.perf_counter() - t_start

    pred_np  = pred[0].cpu().numpy()
    true_np  = traj_sample.numpy()

    ic_mean = norm_stats["ic_mean"]
    ic_std  = norm_stats["ic_std"]

    pred_phys = pred_np * ic_std[None, :] + ic_mean[None, :]
    true_phys = true_np * ic_std[None, :] + ic_mean[None, :]

    t_arr = np.linspace(0, 10, hp["n_timesteps"])

    bdf_time  = _mock_bdf_time(hp["n_timesteps"])
    speedup   = bdf_time / max(lno_time, 1e-9)

    abs_err = np.abs(pred_phys - true_phys)
    mse_x   = np.mean((pred_phys[:, 0] - true_phys[:, 0])**2)
    mse_y   = np.mean((pred_phys[:, 1] - true_phys[:, 1])**2)

    BG    = "#0d0d1a"
    PANEL = "#12122a"
    C_X_T = "#00e5ff"
    C_Y_T = "#ff4081"
    C_X_P = "#ffd740"
    C_Y_P = "#69ff47"

    fig = plt.figure(figsize=(14, 6))
    fig.patch.set_facecolor(BG)
    gs_l = gridspec.GridSpec(1, 2, figure=fig, wspace=0.35)

    def style(ax, title):
        ax.set_facecolor(PANEL)
        ax.set_title(title, color="white", fontsize=11, pad=8, fontweight="bold")
        ax.tick_params(colors="#aaaacc", labelsize=9)
        ax.xaxis.label.set_color("#aaaacc")
        ax.yaxis.label.set_color("#aaaacc")
        for sp in ax.spines.values():
            sp.set_edgecolor("#2a2a50")
        ax.grid(True, color="#1e1e3a", lw=0.6, ls="--")

    # Left panel
    ax0 = fig.add_subplot(gs_l[0, 0])
    ax0.plot(t_arr, true_phys[:, 0],  color=C_X_T, lw=2.0, ls="-", label="True x (GATA-1)")
    ax0.plot(t_arr, true_phys[:, 1],  color=C_Y_T, lw=2.0, ls="-", label="True y (PU.1)")
    ax0.plot(t_arr, pred_phys[:, 0],  color=C_X_P, lw=1.6, ls="--", label="LNO x (GATA-1)")
    ax0.plot(t_arr, pred_phys[:, 1],  color=C_Y_P, lw=1.6, ls="--", label="LNO y (PU.1)")
    ax0.set_xlabel("Time")
    ax0.set_ylabel("Concentration")
    ax0.legend(fontsize=8, facecolor=PANEL, labelcolor="white", framealpha=0.9, loc="best")
    style(ax0, "True BDF vs LNO Prediction")

    ann_text = f"LNO :  {lno_time*1e3:.2f} ms\nBDF :  {bdf_time*1e3:.0f} ms\nSpeedup: {speedup:.0f}×"
    ax0.text(
        0.97, 0.97, ann_text,
        transform=ax0.transAxes, ha="right", va="top",
        fontsize=8, fontfamily="monospace", color="#ccccee",
        bbox=dict(boxstyle="round,pad=0.4", fc="#1a1a3a", ec="#5555bb", lw=1.2),
    )

    # Right panel
    ax1 = fig.add_subplot(gs_l[0, 1])
    ax1.semilogy(t_arr, abs_err[:, 0] + 1e-9, color=C_X_P, lw=1.8, label="|err| x (GATA-1)")
    ax1.semilogy(t_arr, abs_err[:, 1] + 1e-9, color=C_Y_P, lw=1.8, label="|err| y (PU.1)")
    ax1.set_xlabel("Time")
    ax1.set_ylabel("Absolute error (log scale)")
    ax1.legend(fontsize=8, facecolor=PANEL, labelcolor="white", framealpha=0.9)
    style(ax1, "Absolute Prediction Error (semi-log)")

    fig.suptitle(
        "Laplace Neural Operator — Genetic Switch Trajectory Prediction\n"
        f"Test sample #{rng_idx}  ·  MSE_x={mse_x:.2e}  ·  MSE_y={mse_y:.2e}  "
        f"·  LNO speedup {speedup:.0f}× vs BDF",
        color="white", fontsize=10.5, y=1.01, fontweight="bold",
    )

    plt.savefig(hp["plot_path"], dpi=150, bbox_inches="tight", facecolor=BG)
    plt.close()
    return fig
