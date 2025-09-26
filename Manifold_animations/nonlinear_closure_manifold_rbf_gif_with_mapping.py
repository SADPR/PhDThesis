#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Nonlinear closure manifold (Φ-based map) with RBF interpolation.
Animated GIF version — same flow as ANN:
  trajectory -> linear approx -> fade-in manifold -> grow nonlinear closure curve
  with per-frame "lift" -> hold -> spin.
"""

# ================== IMPORTS ==================
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ================== STYLE ====================
plt.rcParams.update({
    "text.usetex": True,         # set False if LaTeX unavailable
    "font.family": "serif",
    "axes.titlesize": 20,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

# ================== CONFIG ===================
GIF_NAME          = "closure_manifold_mathcalNphi.gif"
FPS               = 24
SURF_ALPHA        = 0.15
GRID_ALPHA        = 0.35
SURF_FADE_FRAMES  = 20
SPIN_FRAMES       = 240
HOLD_FRAMES       = 24
ELEV0, AZIM0      = 15, 225
SEED              = 123

np.random.seed(SEED)

# ============== 1) Trajectory s(t) ===========
t = np.linspace(0, 2*np.pi, 100)
S = np.c_[np.cos(t), np.sin(t), 0.5*np.cos(2*t)].astype(float)  # (N,3)
u_ref = S[0]
S_shift = S - u_ref

# ============== 2) POD basis ==================
U, svals, VT = np.linalg.svd(S_shift, full_matrices=False)
V   = VT.T
V_p = V[:, :2]                # (3x2) kept plane
V_s = V[:,  2:3]              # (3x1) discarded (closure) direction

# ============== 3) Reduced coords =============
q_p = S_shift @ V_p           # (N,2)
q_s = (S_shift @ V_s).ravel() # (N,)

# ============== 4) Closure map (RBF) ==========
rbf = Rbf(q_p[:,0], q_p[:,1], q_s, function='multiquadric', smooth=0.0)
q_s_pred = rbf(q_p[:,0], q_p[:,1])

# 2D (plane) approximation and nonlinear closure reconstruction
S_2d  = u_ref + q_p @ V_p.T
S_3d  = u_ref + q_p @ V_p.T + np.outer(q_s_pred, V_s.ravel())

# ============== 5) Surface grid in latent =====
z1 = np.linspace(q_p[:,0].min() - 0.2, q_p[:,0].max() + 0.2, 35)
z2 = np.linspace(q_p[:,1].min() - 0.2, q_p[:,1].max() + 0.2, 35)
Z1, Z2 = np.meshgrid(z1, z2)

qs_grid = rbf(Z1, Z2)

# Map graph (q_p, N_Phi(q_p)) back to ambient R^3
Surf_lin  = (V_p @ np.stack([Z1.ravel(), Z2.ravel()], axis=0)).reshape(3, *Z1.shape)
Surf_full = u_ref[:, None, None] + Surf_lin + (V_s @ qs_grid.ravel()[None, :]).reshape(3, *Z1.shape)
X, Y, Z   = Surf_full

# ============== 6) Animation setup ============
fig = plt.figure(figsize=(9,7))
ax  = fig.add_subplot(111, projection='3d')
ax.set_facecolor("white")
fig.patch.set_alpha(1.0)

ax.set_xlim([-1.5, 1.5]); ax.set_ylim([-1.5, 1.5]); ax.set_zlim([-0.6, 0.6])
ax.set_xlabel(r'$s_1$'); ax.set_ylabel(r'$s_2$'); ax.set_zlabel(r'$s_3$')
ax.set_title(r'\textbf{Nonlinear closure manifold} $(\mathcal{N}_{\Phi})$', pad=50)
ax.view_init(elev=ELEV0, azim=AZIM0)

# Artists
pts_line, = ax.plot([], [], [], 'o', color='black', markersize=3,
                    label=r'\textit{trajectory} $\mathbf{s}(t)$')

u2d_line, = ax.plot([], [], [], linestyle='-', color='gray', lw=1.5,
                    label=r'linear approximation'+'\n'+r'\hspace{1em}$(\mathbf{\Phi}\,\mathbf{q})$')

u3d_line, = ax.plot([], [], [], linestyle='--', color='forestgreen', lw=2.2,
                    label=r'nonlinear closure approximation'+'\n'
                          + r'\hspace{1em}$(\mathbf{\Phi}\,\mathbf{q} + \mathbf{\bar{\Phi}}\,\mathcal{N}_{\Phi}(\mathbf{q}))$')

# Per-frame lift segment (not in legend)
lift_line, = ax.plot([], [], [], color='darkgreen', lw=2, alpha=0.9)

surf = ax.plot_surface(X, Y, Z, color='darkolivegreen', alpha=0.0,
                       edgecolor='none', lw=0.0, antialiased=True, shade=False)
wire = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2,
                         color='gray', linewidth=0.4, alpha=0.0)

# Legend
legend_handles = [
    Line2D([], [], color='k', marker='o', ls='None',
           label=r'\textit{trajectory} $\mathbf{s}(t)$'),
    Line2D([], [], color='gray', lw=1.8,
           label=r'linear approximation'+'\n'+r'\hspace{1em}$(\mathbf{\Phi}\,\mathbf{q})$'),
    Line2D([], [], color='forestgreen', ls='--', lw=2.2,
           label=r'nonlinear closure approximation'+'\n'
                 + r'\hspace{1em}$(\mathbf{\Phi}\,\mathbf{q} + \mathbf{\bar{\Phi}}\,\mathcal{N}_{\Phi}(\mathbf{q}))$'),
    Patch(facecolor='darkolivegreen', alpha=0.3, edgecolor='gray',
          label=r'nonlinear closure manifold $(\mathcal{N}_{\Phi})$')
]
ax.set_position([0.10, 0.03, 0.80, 0.82])
leg = ax.legend(handles=legend_handles,
                loc='upper right',
                bbox_to_anchor=(0.92, 0.93),
                bbox_transform=fig.transFigure,
                frameon=True, fancybox=True, framealpha=0.6,
                borderpad=0.8, handlelength=2.8, handletextpad=0.8,
                labelspacing=0.6, columnspacing=1.0)

# ============== 7) Timeline ===================
N_pts   = len(S)
N_u2d   = len(S_2d)
N_surf  = SURF_FADE_FRAMES
N_u3d   = len(S_3d)
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
        # Grow 3D curve; show lift
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
