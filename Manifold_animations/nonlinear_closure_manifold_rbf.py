# nonlinear_closure_manifold_rbf.py ----------------------------------------
import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

# === 4. Fit RBF for closure ================================================
rbf = Rbf(q_p[:,0], q_p[:,1], q_s, function='multiquadric', smooth=0.0)
q_s_pred = rbf(q_p[:,0], q_p[:,1])
S_pred = u_ref + q_p @ V_p.T + np.outer(q_s_pred, V_s.ravel())

# === 5. Evaluate surface of manifold =======================================
gp1 = np.linspace(q_p[:,0].min() - 0.15, q_p[:,0].max() + 0.15, 60)
gp2 = np.linspace(q_p[:,1].min() - 0.15, q_p[:,1].max() + 0.15, 60)
GP1, GP2 = np.meshgrid(gp1, gp2)
qs_grid = rbf(GP1, GP2)

# === 6. Map to ambient space ===============================================
U_grid = ( u_ref[:, None, None]
           + (V_p @ np.stack([GP1, GP2]).reshape(2, -1)).reshape(3, *GP1.shape)
           + (V_s @ qs_grid.ravel()[None, :]).reshape(3, *GP1.shape) )

X_surf, Y_surf, Z_surf = U_grid

# === 7. Plot ===============================================================
fig = plt.figure(figsize=(9, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot3D(*S.T, 'ko', ms=4, label=r'\textit{trajectory} $\mathbf{s}(t)$')
ax.plot3D(*S_pred.T, color='forestgreen', lw=2, ls='--', label=r'\textit{approximated trajectory}')
ax.plot_surface(X_surf, Y_surf, Z_surf,
                color='darkolivegreen', alpha=0.25,
                edgecolor='gray', linewidth=0.2)

ax.set(xlim=(-1.5,1.5), ylim=(-1.5,1.5), zlim=(-0.6,0.6),
       xlabel=r'$s_1$', ylabel=r'$s_2$', zlabel=r'$s_3$')
ax.set_title(r'\textbf{Nonlinear closure manifold (RBF-based)}', pad=18)
ax.view_init(elev=15, azim=225)

# === 8. Legend =============================================================
ax.legend(handles=[
    Line2D([],[], color='k', marker='o', ls='None', label=r'trajectory $\mathbf{s}(t)$'),
    Line2D([],[], color='forestgreen', lw=2, ls='--', label=r'approximated trajectory'),
    Patch(facecolor='darkolivegreen', alpha=0.3, edgecolor='gray', label=r'nonlinear closure manifold')
], loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig("nonlinear_closure_manifold_rbf.pdf", format="pdf", bbox_inches='tight')
plt.show()
