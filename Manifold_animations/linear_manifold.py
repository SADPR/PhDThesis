import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

# Global LaTeX plot settings
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

# === 1. Define trajectory s(t) =============================================
t_vals = np.linspace(0, 2 * np.pi, 100)
s1 = np.cos(t_vals)
s2 = np.sin(t_vals)
s3 = 0.5 * np.cos(2 * t_vals)
S = np.stack((s1, s2, s3), axis=1)  # shape (100, 3)

# === 2. Use s(0) as reference ===============================================
s_ref = S[0]
S_shifted = S - s_ref

# === 3. POD basis from SVD =================================================
U, Sigma, VT = np.linalg.svd(S_shifted, full_matrices=False)
V = VT[:2, :].T  # (3x2)

# === 4. Projection and reconstruction =======================================
hat_s = S_shifted @ V
S_linear = s_ref + hat_s @ V.T

# === 5. Grid for manifold surface ===========================================
hat_s_mean = np.mean(hat_s, axis=0)
alpha = np.linspace(-1.5, 1.5, 30) + hat_s_mean[0]
beta = np.linspace(-1.5, 1.5, 30) + hat_s_mean[1]
A, B = np.meshgrid(alpha, beta)
plane_points = s_ref[:, None] + V @ np.vstack((A.ravel(), B.ravel()))
X_plane = plane_points[0].reshape(30, 30)
Y_plane = plane_points[1].reshape(30, 30)
Z_plane = plane_points[2].reshape(30, 30)

# === 6. Plot =================================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Trajectory and approximation
ax.plot(S[:, 0], S[:, 1], S[:, 2], 'o', color='black', markersize=4, label=r'\textit{trajectory} $\mathbf{s}(t)$')
ax.plot(S_linear[:, 0], S_linear[:, 1], S_linear[:, 2], color='dodgerblue', linewidth=2, label=r'\textit{approximated trajectory}')

# Linear manifold surface
ax.plot_surface(X_plane, Y_plane, Z_plane, color='deepskyblue', edgecolor='gray', linewidth=0.3, alpha=0.15)

# Axis labels and view
ax.set_xlim([-1.5, 1.5])
ax.set_ylim([-1.5, 1.5])
ax.set_zlim([-0.6, 0.6])
ax.set_xlabel(r'$s_1$')
ax.set_ylabel(r'$s_2$')
ax.set_zlabel(r'$s_3$')
ax.set_title(r'\textbf{Linear manifold}', pad=20)
ax.view_init(elev=15, azim=225)

# === 7. Classic top-right legend ============================================
legend_elements = [
    Line2D([0], [0], color='black', marker='o', linestyle='None', label=r'trajectory $\mathbf{s}(t)$'),
    Line2D([0], [0], color='dodgerblue', lw=2, label=r'approximated trajectory'),
    Patch(facecolor='deepskyblue', edgecolor='gray', label=r'linear manifold', alpha=0.3)
]
ax.legend(
    handles=legend_elements,
    loc='upper right',
    frameon=True
)

# === 8. Save and show =======================================================
plt.savefig("linear_manifold.pdf", format="pdf", bbox_inches='tight')
plt.show()
