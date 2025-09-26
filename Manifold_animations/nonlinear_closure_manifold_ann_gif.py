#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# ================== IMPORTS ==================
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ================== STYLE ====================
plt.rcParams.update({
    "text.usetex": True,         # set to False if LaTeX is unavailable
    "font.family": "serif",
    "axes.titlesize": 20,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

# ================== CONFIG ===================
GIF_NAME          = "ann_closure_manifold.gif"
FPS               = 24
SURF_ALPHA        = 0.15       # final transparency of surface fill
GRID_ALPHA        = 0.35       # final transparency of wireframe
SURF_FADE_FRAMES  = 20         # fade-in frames for surface + wireframe
SPIN_FRAMES       = 240        # frames for one 360° spin
HOLD_FRAMES       = 24         # short pause before spin
ELEV0, AZIM0      = 15, 225    # initial camera
SEED              = 123

# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_default_dtype(torch.float32)

# ============== 1) Trajectory s(t) ===========
t = np.linspace(0, 2*np.pi, 100)
S = np.c_[np.cos(t), np.sin(t), 0.5*np.cos(2*t)].astype(np.float32)  # (100,3)

# Reference shift (can use mean; here we use first point to match your code)
u_ref = S[0]
S_shift = S - u_ref

# ============== 2) POD basis ==================
U, svals, VT = np.linalg.svd(S_shift, full_matrices=False)
V   = VT.T                   # (3x3), V^T V = I
V_p = V[:, :2]               # primary subspace (3x2)
V_s = V[:,  2:3]             # closure direction (3x1)

# ============== 3) Reduced coords =============
q_p = S_shift @ V_p          # (100,2)
q_s = (S_shift @ V_s).ravel()# (100,)

# ============== 4) ANN closure =================
class ClosureANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ELU(),
            nn.Linear(16, 16),
            nn.ELU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):  # x: (..., 2)
        return self.net(x)

model     = ClosureANN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

X_train = torch.from_numpy(q_p)            # (N,2)
y_train = torch.from_numpy(q_s[:, None])   # (N,1)

for epoch in range(1000):
    optimizer.zero_grad()
    pred = model(X_train)
    loss = criterion(pred, y_train)
    loss.backward()
    optimizer.step()
    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch+1:4d} | Loss: {loss.item():.6e}")

# ============== 5) Predict & reconstruct ======
with torch.no_grad():
    q_s_pred = model(X_train).cpu().numpy().ravel()

S_pred = u_ref + q_p @ V_p.T + np.outer(q_s_pred, V_s.ravel())  # (N,3)

# ============== 6) Surface grid in latent =====
z1 = np.linspace(q_p[:,0].min() - 0.2, q_p[:,0].max() + 0.2, 35)
z2 = np.linspace(q_p[:,1].min() - 0.2, q_p[:,1].max() + 0.2, 35)
Z1, Z2 = np.meshgrid(z1, z2)
grid_qp = np.stack([Z1.ravel(), Z2.ravel()], axis=1).astype(np.float32)  # (M,2)

with torch.no_grad():
    q_s_grid = model(torch.from_numpy(grid_qp)).cpu().numpy().ravel()    # (M,)

# Decode to 3D: u = u_ref + V_p z + V_s q_s(z)
Surf_lin  = (V_p @ grid_qp.T)                           # (3,M)
Surf_full = u_ref[:,None] + Surf_lin + V_s @ q_s_grid[None,:]  # (3,M)
X, Y, Z   = [a.reshape(Z1.shape) for a in Surf_full]    # (35,35) each

# Quick errors (optional)
err_lin  = np.linalg.norm(S - (u_ref + q_p @ V_p.T))
err_ann  = np.linalg.norm(S - S_pred)
print(f"||S - S_lin||_2 = {err_lin:.6f}   ||S - S_ann||_2 = {err_ann:.6f}")

# ============== 7) Animation setup ============
fig = plt.figure(figsize=(9,7))
ax  = fig.add_subplot(111, projection='3d')
ax.set_xlim([-1.5, 1.5]); ax.set_ylim([-1.5, 1.5]); ax.set_zlim([-0.6, 0.6])
ax.set_xlabel(r'$s_1$'); ax.set_ylabel(r'$s_2$'); ax.set_zlabel(r'$s_3$')
ax.set_title(r'\textbf{Nonlinear closure manifold (ANN-based)}', pad=20)
ax.view_init(elev=ELEV0, azim=AZIM0)

# Artists
traj_line,  = ax.plot([], [], [], 'o', color='black', markersize=4,
                      label=r'\textit{trajectory} $\mathbf{s}(t)$')

surf = ax.plot_surface(X, Y, Z, color='orchid', alpha=0.0,
                       edgecolor='none', lw=0.0, antialiased=True, shade=False)
wire = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2,
                         color='gray', linewidth=0.4, alpha=0.0)

recon_line, = ax.plot([], [], [], linestyle='--', color='mediumvioletred', lw=2,
                      label=r'\textit{approximated trajectory}')

# Legend
ax.legend(loc='upper right', handles=[
    Line2D([], [], color='k', marker='o', ls='None',
           label=r'\textit{trajectory} $\mathbf{s}(t)$'),
    Line2D([], [], color='mediumvioletred', ls='--', lw=2,
           label=r'\textit{approximated trajectory}'),
    Patch(facecolor='orchid', alpha=0.3, edgecolor='gray',
          label=r'nonlinear closure manifold')
])

# Timeline
N_traj  = len(S)                 # grow true trajectory
N_surf  = SURF_FADE_FRAMES       # fade-in surface + grid
N_recon = len(S_pred)            # grow ANN reconstruction
N_hold  = HOLD_FRAMES            # pause
N_spin  = SPIN_FRAMES            # spin
N_total = N_traj + N_surf + N_recon + N_hold + N_spin

def init():
    traj_line.set_data([], []); traj_line.set_3d_properties([])
    recon_line.set_data([], []); recon_line.set_3d_properties([])
    surf.set_alpha(0.0); wire.set_alpha(0.0)
    ax.view_init(elev=ELEV0, azim=AZIM0)
    return traj_line, recon_line, surf, wire

def update(frame):
    if frame < N_traj:
        # 1) draw true trajectory
        k = frame + 1
        traj_line.set_data(S[:k,0], S[:k,1])
        traj_line.set_3d_properties(S[:k,2])

    elif frame < N_traj + N_surf:
        # 2) fade-in manifold surface + wire
        a = (frame - N_traj + 1) / N_surf
        surf.set_alpha(SURF_ALPHA * a)
        wire.set_alpha(GRID_ALPHA * a)

    elif frame < N_traj + N_surf + N_recon:
        # 3) draw ANN reconstruction
        k = frame - (N_traj + N_surf) + 1
        recon_line.set_data(S_pred[:k,0], S_pred[:k,1])
        recon_line.set_3d_properties(S_pred[:k,2])
        surf.set_alpha(SURF_ALPHA)
        wire.set_alpha(GRID_ALPHA)

    elif frame < N_traj + N_surf + N_recon + N_hold:
        # 4) hold
        pass

    else:
        # 5) spin
        k = frame - (N_traj + N_surf + N_recon + N_hold)
        frac = k / max(1, N_spin - 1)
        ax.view_init(elev=ELEV0, azim=AZIM0 + 360.0 * frac)

    return traj_line, recon_line, surf, wire

ani = FuncAnimation(fig, update, frames=N_total, init_func=init, blit=False)
ani.save(GIF_NAME, writer=PillowWriter(fps=FPS))
plt.close(fig)
print(f"Saved GIF to: {GIF_NAME}")
