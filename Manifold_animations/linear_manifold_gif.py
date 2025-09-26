import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# ---------- CONFIG ----------
GIF_NAME = "linear_manifold.gif"
PLANE_TARGET_ALPHA = 0.15
HOLD_FRAMES = 24         # static pause before spin
SPIN_FRAMES = 240        # frames for one full 360° spin
# ----------------------------

# Match your LaTeX style and fonts
plt.rcParams.update({
    'text.usetex': True,
    'font.family': 'serif',
    'axes.titlesize': 20,
    'axes.labelsize': 14,
    'xtick.labelsize': 14,
    'ytick.labelsize': 14,
    'legend.fontsize': 14,
    'figure.autolayout': False
})

# === 1) Define trajectory s(t)
t_vals = np.linspace(0, 2 * np.pi, 150)  # dense for smooth animation
s1 = np.cos(t_vals)
s2 = np.sin(t_vals)
s3 = 0.5 * np.cos(2 * t_vals)
S = np.stack((s1, s2, s3), axis=1)  # (N,3)

# === 2) Use s(0) as reference (your initial condition)
s_ref = S[0]
S_shifted = S - s_ref

# === 3) POD basis from SVD (rank-2)
U, Sigma, VT = np.linalg.svd(S_shifted, full_matrices=False)
V = VT[:2, :].T  # (3x2)

# === 4) Projection and reconstruction
hat_s = S_shifted @ V
S_linear = s_ref + hat_s @ V.T

# === 5) Plane grid (same palette)
hat_s_mean = np.mean(hat_s, axis=0)
alpha = np.linspace(-1.5, 1.5, 30) + hat_s_mean[0]
beta  = np.linspace(-1.5, 1.5, 30) + hat_s_mean[1]
A, B = np.meshgrid(alpha, beta)
plane_points = s_ref[:, None] + V @ np.vstack((A.ravel(), B.ravel()))
X_plane = plane_points[0].reshape(30, 30)
Y_plane = plane_points[1].reshape(30, 30)
Z_plane = plane_points[2].reshape(30, 30)

# === 6) Figure + artists
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Axes/labels/title/view (match yours)
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-0.6, 0.6])
ax.set_xlabel(r'$s_1$')
ax.set_ylabel(r'$s_2$')
ax.set_zlabel(r'$s_3$')
ax.set_title(r'\textbf{Linear manifold}', pad=20)
elev0, azim0 = 15, 225
ax.view_init(elev=elev0, azim=azim0)

# Trajectory (black dots) and reconstruction (dodgerblue line)
traj_line,  = ax.plot([], [], [], 'o', color='black', markersize=4,
                      label=r'\textit{trajectory} $\mathbf{s}(t)$')
recon_line, = ax.plot([], [], [], color='dodgerblue', linewidth=2,
                      label=r'\textit{approximated trajectory}')

# Plane surface (start fully hidden)
surf = ax.plot_surface(X_plane, Y_plane, Z_plane,
                       color='deepskyblue', edgecolor='gray',
                       linewidth=0.3, alpha=0.0)
surf.set_visible(False)   # << fully invisible until fade-in starts

# Legend (your style)
legend_elements = [
    Line2D([0], [0], color='black', marker='o', linestyle='None',
           label=r'trajectory $\mathbf{s}(t)$'),
    Line2D([0], [0], color='dodgerblue', lw=2, label=r'approximated trajectory'),
    Patch(facecolor='deepskyblue', edgecolor='gray', label=r'linear manifold', alpha=0.3)
]
ax.legend(handles=legend_elements, loc='upper right', frameon=True)

# === 7) Animation phases
N = len(S)
N_TRAJ  = N                  # grow trajectory
N_PLANE = 20                 # fade-in frames
N_RECON = N                  # grow reconstruction
N_HOLD  = HOLD_FRAMES        # static pause
N_SPIN  = SPIN_FRAMES        # one full spin

N_total = N_TRAJ + N_PLANE + N_RECON + N_HOLD + N_SPIN

def init():
    traj_line.set_data([], [])
    traj_line.set_3d_properties([])
    recon_line.set_data([], [])
    recon_line.set_3d_properties([])
    surf.set_alpha(0.0)
    surf.set_visible(False)
    ax.view_init(elev=elev0, azim=azim0)
    return traj_line, recon_line, surf

def update(frame):
    if frame < N_TRAJ:
        # Phase 1: draw trajectory points
        k = frame + 1
        traj_line.set_data(S[:k, 0], S[:k, 1])
        traj_line.set_3d_properties(S[:k, 2])

    elif frame < N_TRAJ + N_PLANE:
        # Phase 2: fade-in plane (not visible before this)
        if not surf.get_visible():
            surf.set_visible(True)
        a = (frame - N_TRAJ + 1) / N_PLANE
        surf.set_alpha(PLANE_TARGET_ALPHA * a)

    elif frame < N_TRAJ + N_PLANE + N_RECON:
        # Phase 3: draw reconstructed trajectory
        k = frame - (N_TRAJ + N_PLANE) + 1
        recon_line.set_data(S_linear[:k, 0], S_linear[:k, 1])
        recon_line.set_3d_properties(S_linear[:k, 2])
        surf.set_alpha(PLANE_TARGET_ALPHA)

    elif frame < N_TRAJ + N_PLANE + N_RECON + N_HOLD:
        # Phase 4: hold (static, no spin)
        pass

    else:
        # Phase 5: single spin around s3 (z) after everything is shown
        k = frame - (N_TRAJ + N_PLANE + N_RECON + N_HOLD)
        frac = k / max(1, N_SPIN - 1)
        ax.view_init(elev=elev0, azim=azim0 + 360.0 * frac)

    return traj_line, recon_line, surf

ani = FuncAnimation(fig, update, frames=N_total, init_func=init, blit=False)
ani.save(GIF_NAME, writer=PillowWriter(fps=24))
plt.close(fig)
print(f"Saved GIF to: {GIF_NAME}")
