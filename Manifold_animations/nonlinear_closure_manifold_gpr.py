import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern, ConstantKernel as C

# Global LaTeX plot settings
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.titlesize": 20,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

# === 1. Define trajectory s(t) =============================================
t = np.linspace(0, 2*np.pi, 100)
S = np.c_[np.cos(t), np.sin(t), 0.5*np.cos(2*t)].astype(float)  # shape (100, 3)
u_ref = S[0]
S_shift = S - u_ref

# === 2. POD basis decomposition ============================================
U, svals, VT = np.linalg.svd(S_shift, full_matrices=False)
V = VT.T
V_p = V[:, :2]        # primary basis (3x2)
V_s = V[:,  2:3]      # secondary (closure) basis (3x1)

# === 3. Reduced coordinates ================================================
q_p = S_shift @ V_p                 # shape (100, 2)
q_s = (S_shift @ V_s).ravel()       # shape (100,)

# === 4. Fit GPR model for closure ==========================================
kernel = C(1.0, (1e-2, 1e2)) * Matern(nu=1.5)
gpr = GaussianProcessRegressor(kernel=kernel, alpha=1e-6, normalize_y=True)
gpr.fit(q_p, q_s)
print("GPR fitted kernel:", gpr.kernel_)

# === 5. Predict and reconstruct ============================================
q_s_pred = gpr.predict(q_p)
S_pred = u_ref + q_p @ V_p.T + np.outer(q_s_pred, V_s.ravel())

# === 6. Evaluate surface of the manifold ===================================
gp1 = np.linspace(q_p[:,0].min() - 0.15, q_p[:,0].max() + 0.15, 60)
gp2 = np.linspace(q_p[:,1].min() - 0.15, q_p[:,1].max() + 0.15, 60)
GP1, GP2 = np.meshgrid(gp1, gp2)
qs_grid = gpr.predict(np.c_[GP1.ravel(), GP2.ravel()]).reshape(GP1.shape)

# === 7. Map to ambient space ===============================================
V_part = V_p @ np.stack([GP1, GP2]).reshape(2, -1)
W_part = V_s @ qs_grid.ravel()[None, :]
U_grid = u_ref[:, None] + V_part + W_part
X_surf, Y_surf, Z_surf = U_grid.reshape(3, *GP1.shape)

# === 8. Plot ===============================================================
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(*S.T, 'ko', ms=4, label=r'\textit{trajectory} $\mathbf{s}(t)$')
ax.plot3D(*S_pred.T, color='midnightblue', lw=2, ls='--', label=r'\textit{approximated trajectory}')
ax.plot_surface(X_surf, Y_surf, Z_surf,
                color='steelblue', alpha=0.25,
                edgecolor='gray', linewidth=0.2)

ax.set(xlim=(-1.5,1.5), ylim=(-1.5,1.5), zlim=(-0.6,0.6),
       xlabel=r'$s_1$', ylabel=r'$s_2$', zlabel=r'$s_3$')
ax.set_title(r'\textbf{Nonlinear closure manifold (GPR-based)}', pad=18)
ax.view_init(elev=15, azim=225)

# === 9. Legend =============================================================
ax.legend(handles=[
    Line2D([],[], color='k', marker='o', ls='None', label=r'trajectory $\mathbf{s}(t)$'),
    Line2D([],[], color='midnightblue', lw=2, ls='--', label=r'approximated trajectory'),
    Patch(facecolor='steelblue', alpha=0.3, edgecolor='gray', label=r'nonlinear closure manifold')
], loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig("nonlinear_closure_manifold_gpr.pdf", format="pdf", bbox_inches='tight')
plt.show()

# === 10. ------------------- ADDITIONAL PLOTS: Uncertainty ====================

# Predict both mean and std over the grid
grid_points = np.c_[GP1.ravel(), GP2.ravel()]
qs_mu, qs_std = gpr.predict(grid_points, return_std=True)

# Compute upper and lower 95% confidence bounds
qs_upper = qs_mu + 1.96 * qs_std
qs_lower = qs_mu - 1.96 * qs_std

# Define helper to convert latent (q_p, q_s) to physical space
def map_to_surface(qs_flat):
    V_part = V_p @ grid_points.T  # (3 x N)
    W_part = V_s @ qs_flat.reshape(1, -1)
    U_grid = u_ref[:, None] + V_part + W_part  # (3 x N)
    return U_grid.reshape(3, *GP1.shape)

X_mean, Y_mean, Z_mean = map_to_surface(qs_mu)
X_upper, Y_upper, Z_upper = map_to_surface(qs_upper)
X_lower, Y_lower, Z_lower = map_to_surface(qs_lower)

# === 11. Plot uncertainty bounds as surfaces =================================
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot3D(*S.T, 'ko', ms=4, label=r'\textit{trajectory} $\mathbf{s}(t)$')
ax.plot3D(*S_pred.T, color='midnightblue', lw=2, ls='--', label=r'\textit{approximated trajectory}')
ax.plot_surface(X_mean, Y_mean, Z_mean, color='steelblue', alpha=0.25, edgecolor='gray', linewidth=0.2)

# Plot confidence surfaces
ax.plot_surface(X_upper, Y_upper, Z_upper, color='gray', alpha=0.15, linewidth=0, edgecolor='none')
ax.plot_surface(X_lower, Y_lower, Z_lower, color='gray', alpha=0.15, linewidth=0, edgecolor='none')

# Axes and title
ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-0.6, 0.6),
       xlabel=r'$s_1$', ylabel=r'$s_2$', zlabel=r'$s_3$')
ax.set_title(r'\textbf{GPR-based manifold with 95\% confidence bounds}', pad=20)
ax.view_init(elev=15, azim=225)

# Legend
ax.legend(handles=[
    Line2D([], [], color='k', marker='o', ls='None', label=r'trajectory $\mathbf{s}(t)$'),
    Line2D([], [], color='midnightblue', lw=2, ls='--', label=r'approximated trajectory'),
    Patch(facecolor='steelblue', alpha=0.3, edgecolor='gray', label=r'mean prediction'),
    Patch(facecolor='gray', alpha=0.15, edgecolor='none', label=r'95\% confidence bounds')
], loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig("nonlinear_closure_manifold_gpr_confidence_bounds.pdf", format="pdf", bbox_inches='tight')
plt.show()




