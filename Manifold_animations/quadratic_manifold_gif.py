#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# -------------------- STYLE --------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.titlesize": 20,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

# -------------------- CONFIG -------------------
GIF_NAME = "quadratic_manifold.gif"
FPS = 24
SURF_ALPHA = 0.15         # target transparency of filled surface
GRID_ALPHA = 0.35         # target transparency of wireframe grid
SPIN_FRAMES = 240         # frames for one final 360° spin
SURF_FADE_FRAMES = 20     # fade-in frames for the surface & grid

# ----------------- 1) SNAPSHOTS ----------------
t = np.linspace(0, 2*np.pi, 100)
S = np.c_[np.cos(t), np.sin(t), 0.5*np.cos(2*t)]  # (100,3)

# Reference (mean-centering recommended for conditioning)
s_ref = S.mean(axis=0)
S_shift = S - s_ref

# ----------------- 2) POD (rank-2) --------------
U, sing_vals, VT = np.linalg.svd(S_shift, full_matrices=False)
V = VT[:2].T                            # (3x2) with V^T V = I
z = S_shift @ V                         # (N,2)
S_lin = z @ V.T                         # (N,3)
R = S_shift - S_lin                     # residuals

# --------------- 3) Quadratic map ---------------
# q = [z1^2, z1*z2, z2^2] in R^3
Q = np.c_[z[:,0]**2, z[:,0]*z[:,1], z[:,1]**2]   # (N,3)

gamma = 0.0                                       # ridge regularization (0 = none)
A = Q.T @ Q + gamma*np.eye(3)
B = Q.T @ R
W = np.linalg.solve(A, B).T                       # (3x3)

# --------------- 4) Reconstruction --------------
S_quad = s_ref + S_lin + Q @ W.T                  # (N,3)

# --------------- 5) Manifold surface ------------
z1_min, z1_max = z[:,0].min() - 0.1, z[:,0].max() + 0.1
z2_min, z2_max = z[:,1].min() - 0.1, z[:,1].max() + 0.1
g1 = np.linspace(z1_min, z1_max, 35)
g2 = np.linspace(z2_min, z2_max, 35)
G1, G2 = np.meshgrid(g1, g2)
Zgrid  = np.vstack((G1.ravel(), G2.ravel()))      # (2,M)

Surf_lin  = V @ Zgrid
Qgrid = np.vstack((Zgrid[0]**2, Zgrid[0]*Zgrid[1], Zgrid[1]**2))  # (3,M)
Surf_full = s_ref[:,None] + Surf_lin + W @ Qgrid                   # (3,M)
X, Y, Z = [a.reshape(G1.shape) for a in Surf_full]

# (Optional) quick error print
err_lin  = np.linalg.norm(S - (s_ref + S_lin),  2)
err_quad = np.linalg.norm(S - S_quad,          2)
print(f"||S - S_lin||_2  = {err_lin:.6f}   ||S - S_quad||_2 = {err_quad:.6f}")

# --------------- 6) Animation setup -------------
fig = plt.figure(figsize=(9,7))
ax  = fig.add_subplot(111, projection='3d')

ax.set_xlim([-1.5, 1.5]); ax.set_ylim([-1.5, 1.5]); ax.set_zlim([-0.6, 0.6])
ax.set_xlabel(r'$s_1$'); ax.set_ylabel(r'$s_2$'); ax.set_zlabel(r'$s_3$')
ax.set_title(r'\textbf{Quadratic manifold}', pad=20)
elev0, azim0 = 15, 225
ax.view_init(elev=elev0, azim=azim0)

# Artists
traj_line,  = ax.plot([], [], [], 'o', color='black', markersize=4,
                      label=r'\textit{trajectory} $\mathbf{s}(t)$')

# Filled surface (plum) + wireframe grid (gray) — both fade in
surf = ax.plot_surface(X, Y, Z, color='plum', alpha=0.0,
                       edgecolor='none', lw=0.0, antialiased=True, shade=False)
wire = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2,
                         color='gray', linewidth=0.4, alpha=0.0)

recon_line, = ax.plot([], [], [], color='magenta', lw=2,
                      label=r'\textit{approximated trajectory}')

# Legend (static)
ax.legend(loc='upper right', handles=[
    Line2D([], [], color='k', marker='o', ls='None', label=r'\textit{trajectory} $\mathbf{s}(t)$'),
    Line2D([], [], color='magenta', lw=2, label=r'\textit{approximated trajectory}'),
    Patch(facecolor='plum', alpha=0.3, edgecolor='gray', label=r'quadratic manifold')
])

# --------------- 7) Timeline --------------------
N_traj  = len(S)              # grow trajectory points
N_surf  = SURF_FADE_FRAMES    # fade-in surface + grid
N_recon = len(S)              # grow reconstructed trajectory
N_hold  = 24                  # short pause
N_spin  = SPIN_FRAMES         # one full spin

N_total = N_traj + N_surf + N_recon + N_hold + N_spin

def init():
    traj_line.set_data([], []); traj_line.set_3d_properties([])
    recon_line.set_data([], []); recon_line.set_3d_properties([])
    surf.set_alpha(0.0)
    wire.set_alpha(0.0)
    ax.view_init(elev=elev0, azim=azim0)
    return traj_line, recon_line, surf, wire

def update(frame):
    if frame < N_traj:
        # Phase 1: draw trajectory
        k = frame + 1
        traj_line.set_data(S[:k,0], S[:k,1])
        traj_line.set_3d_properties(S[:k,2])

    elif frame < N_traj + N_surf:
        # Phase 2: fade-in surface + grid
        a = (frame - N_traj + 1) / N_surf
        surf.set_alpha(SURF_ALPHA * a)
        wire.set_alpha(GRID_ALPHA * a)

    elif frame < N_traj + N_surf + N_recon:
        # Phase 3: draw reconstructed trajectory
        k = frame - (N_traj + N_surf) + 1
        recon_line.set_data(S_quad[:k,0], S_quad[:k,1])
        recon_line.set_3d_properties(S_quad[:k,2])
        surf.set_alpha(SURF_ALPHA)
        wire.set_alpha(GRID_ALPHA)

    elif frame < N_traj + N_surf + N_recon + N_hold:
        # Phase 4: hold (static)
        pass

    else:
        # Phase 5: one 360° spin around z-axis
        k = frame - (N_traj + N_surf + N_recon + N_hold)
        frac = k / max(1, N_spin - 1)
        ax.view_init(elev=elev0, azim=azim0 + 360.0 * frac)

    return traj_line, recon_line, surf, wire

ani = FuncAnimation(fig, update, frames=N_total, init_func=init, blit=False)
ani.save(GIF_NAME, writer=PillowWriter(fps=FPS))
plt.close(fig)
print(f"Saved GIF to: {GIF_NAME}")
