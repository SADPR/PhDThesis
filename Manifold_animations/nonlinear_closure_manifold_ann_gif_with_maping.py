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
GIF_NAME          = "ann_closure_manifold_2d_to_3d.gif"
FPS               = 24
SURF_ALPHA        = 0.15
GRID_ALPHA        = 0.35
SURF_FADE_FRAMES  = 20
SPIN_FRAMES       = 240
HOLD_FRAMES       = 24
ELEV0, AZIM0      = 15, 225
SEED              = 123

np.random.seed(SEED)
torch.manual_seed(SEED)
torch.set_default_dtype(torch.float32)

# ============== 1) Trajectory s(t) ===========
t = np.linspace(0, 2*np.pi, 100)
S = np.c_[np.cos(t), np.sin(t), 0.5*np.cos(2*t)].astype(np.float32)  # (100,3)
u_ref = S[0]
S_shift = S - u_ref

# ============== 2) POD basis ==================
U, svals, VT = np.linalg.svd(S_shift, full_matrices=False)
V   = VT.T
V_p = V[:, :2]                # (3x2)
V_s = V[:,  2:3]              # (3x1)

# ============== 3) Reduced coords =============
q_p = S_shift @ V_p           # (N,2)
q_s = (S_shift @ V_s).ravel() # (N,)

# ============== 4) ANN closure =================
class ClosureANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16), nn.ELU(),
            nn.Linear(16, 16), nn.ELU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):  # x: (..., 2)
        return self.net(x)

model     = ClosureANN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=1e-2)

X_train = torch.from_numpy(q_p)
y_train = torch.from_numpy(q_s[:, None])

for epoch in range(900):
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

# 2D approximation and 3D (ANN) reconstruction
S_2d  = u_ref + q_p @ V_p.T
S_3d  = u_ref + q_p @ V_p.T + np.outer(q_s_pred, V_s.ravel())

# ============== 6) Surface grid in latent =====
z1 = np.linspace(q_p[:,0].min() - 0.2, q_p[:,0].max() + 0.2, 35)
z2 = np.linspace(q_p[:,1].min() - 0.2, q_p[:,1].max() + 0.2, 35)
Z1, Z2 = np.meshgrid(z1, z2)
grid_qp = np.stack([Z1.ravel(), Z2.ravel()], axis=1).astype(np.float32)

with torch.no_grad():
    q_s_grid = model(torch.from_numpy(grid_qp)).cpu().numpy().ravel()

Surf_lin  = (V_p @ grid_qp.T)
Surf_full = u_ref[:,None] + Surf_lin + V_s @ q_s_grid[None,:]
X, Y, Z   = [a.reshape(Z1.shape) for a in Surf_full]

# ============== 7) Animation setup ============
fig = plt.figure(figsize=(9,7))
ax  = fig.add_subplot(111, projection='3d')
ax.set_facecolor("white")            # axes bg
fig.patch.set_alpha(1.0) 
# shift the axes area down a bit
ax.set_xlim([-1.5, 1.5]); ax.set_ylim([-1.5, 1.5]); ax.set_zlim([-0.6, 0.6])
ax.set_xlabel(r'$s_1$'); ax.set_ylabel(r'$s_2$'); ax.set_zlabel(r'$s_3$')
ax.set_title(r'\textbf{Nonlinear closure manifold (ANN-based)}', pad=50)
ax.view_init(elev=ELEV0, azim=AZIM0)

# Artists
pts_line, = ax.plot([], [], [], 'o', color='black', markersize=3,
                    label=r'\textit{trajectory} $\mathbf{s}(t)$')

u2d_line, = ax.plot([], [], [], linestyle='-', color='gray', lw=1.5,
                    label=r'linear approximation'+'\n'+r'\hspace{1em}$(\mathbf{\Phi} \mathbf{q})$')

u3d_line, = ax.plot([], [], [], linestyle='--', color='mediumvioletred', lw=2.2,
                    label=r'nonlinear closure approximation'+'\n'
                          + r'\hspace{1em}$(\mathbf{\Phi} \mathbf{q} + \mathbf{\bar{\Phi}}\,\mathcal{N}(\mathbf{q}))$')

# keep the visual lift but do not include it in the legend
lift_line, = ax.plot([], [], [], color='crimson', lw=2, alpha=0.9)

surf = ax.plot_surface(X, Y, Z, color='orchid', alpha=0.0,
                       edgecolor='none', lw=0.0, antialiased=True, shade=False)
wire = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2,
                         color='gray', linewidth=0.4, alpha=0.0)

# Legend (trajectory + two-line entries + manifold surface)
legend_handles = [
    Line2D([], [], color='k', marker='o', ls='None',
           label=r'\textit{trajectory} $\mathbf{s}(t)$'),
    Line2D([], [], color='gray', lw=1.8,
           label=r'linear approximation'+'\n'+r'\hspace{1em}$(\mathbf{\Phi} \mathbf{q})$'),
    Line2D([], [], color='mediumvioletred', ls='--', lw=2.2,
           label=r'nonlinear closure approximation'+'\n'
                 + r'\hspace{1em}$(\mathbf{\Phi} \mathbf{q} + \mathbf{\bar{\Phi}}\,\mathcal{N}(\mathbf{q}))$'),
    Patch(facecolor='orchid', alpha=0.3, edgecolor='gray',
          label=r'nonlinear closure manifold')
]

# 1) Move only the plotting area downward (left, bottom, width, height)
ax.set_position([0.10, 0.03, 0.80, 0.82])   # tweak bottom/height to taste

# 2) Place the legend in figure coordinates so it doesn't move with the axes
leg = ax.legend(handles=legend_handles,
                loc='upper right',
                bbox_to_anchor=(0.92, 0.93),   # ~top-right of the figure
                bbox_transform=fig.transFigure, # <-- key: pin to figure, not axes
                frameon=True, fancybox=True, framealpha=0.6,
                borderpad=0.8, handlelength=2.8, handletextpad=0.8,
                labelspacing=0.6, columnspacing=1.0)


# Timeline
N_pts   = len(S)               # draw raw data
N_u2d   = len(S_2d)            # grow 2D curve
N_surf  = SURF_FADE_FRAMES     # fade-in manifold
N_u3d   = len(S_3d)            # grow 3D curve + lift
N_hold  = HOLD_FRAMES
N_spin  = SPIN_FRAMES
N_total = N_pts + N_u2d + N_surf + N_u3d + N_hold + N_spin

def init():
    for ln in (pts_line, u2d_line, u3d_line, lift_line):
        ln.set_data([], []); ln.set_3d_properties([])
    surf.set_alpha(0.0); wire.set_alpha(0.0)
    ax.view_init(elev=ELEV0, azim=AZIM0)
    return pts_line, u2d_line, u3d_line, lift_line, surf, wire

def update(frame):
    if frame < N_pts:
        k = frame + 1
        pts_line.set_data(S[:k,0], S[:k,1])
        pts_line.set_3d_properties(S[:k,2])

    elif frame < N_pts + N_u2d:
        k = frame - N_pts + 1
        u2d_line.set_data(S_2d[:k,0], S_2d[:k,1])
        u2d_line.set_3d_properties(S_2d[:k,2])

    elif frame < N_pts + N_u2d + N_surf:
        a = (frame - (N_pts + N_u2d) + 1) / N_surf
        surf.set_alpha(SURF_ALPHA * a)
        wire.set_alpha(GRID_ALPHA * a)

    elif frame < N_pts + N_u2d + N_surf + N_u3d:
        # Grow 3D curve; show lift from current 2D point to 3D point
        k = frame - (N_pts + N_u2d + N_surf) + 1
        u3d_line.set_data(S_3d[:k,0], S_3d[:k,1])
        u3d_line.set_3d_properties(S_3d[:k,2])

        i = max(0, k-1)
        xseg = [S_2d[i,0], S_3d[i,0]]
        yseg = [S_2d[i,1], S_3d[i,1]]
        zseg = [S_2d[i,2], S_3d[i,2]]
        lift_line.set_data(xseg, yseg)
        lift_line.set_3d_properties(zseg)

        surf.set_alpha(SURF_ALPHA); wire.set_alpha(GRID_ALPHA)

    elif frame < N_total - N_spin:
        pass  # hold

    else:
        k = frame - (N_total - N_spin)
        frac = k / max(1, N_spin - 1)
        ax.view_init(elev=ELEV0, azim=AZIM0 + 360.0 * frac)

    return pts_line, u2d_line, u3d_line, lift_line, surf, wire

ani = FuncAnimation(fig, update, frames=N_total, init_func=init, blit=False)
ani.save(GIF_NAME, writer=PillowWriter(fps=FPS))
plt.close(fig)
print(f"Saved GIF to: {GIF_NAME}")

