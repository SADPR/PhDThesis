import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ========== 0.  global figure style =========================================
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.titlesize": 20,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

# ========== 1.  generate snapshots s(t) =====================================
t = np.linspace(0, 2*np.pi, 100)
S = np.c_[np.cos(t), np.sin(t), 0.5*np.cos(2*t)]          # (100,3)

# ------- choose reference state --------------------------------------------
#   mean  → good conditioning
#   S[0]  → poor conditioning
s_ref = S.mean(axis=0)          # try also  S[0]
S_shift = S - s_ref             # centred snapshots

# ========== 2.  linear POD (rank-2) ========================================
U, sing_vals, VT = np.linalg.svd(S_shift, full_matrices=False)
V = VT[:2].T                    # (3×2) basis   V^T V = I_2

# orthogonal projection onto the plane
z      = S_shift @ V            # reduced coordinates  z_i ∈ ℝ²
S_lin  = z @ V.T                # back to ℝ³ (VVᵀ S_shift)

# residual lives in the orthogonal complement of V
R = S_shift - S_lin             # (I – VVᵀ) S_shift   →  (100,3)

# ========== 3.  quadratic mapping  R ≈ Q Wᵀ ================================
# build Kronecker features   q = [z1², z1·z2, z2²]  ∈ ℝ³
Q = np.c_[z[:,0]**2, z[:,0]*z[:,1], z[:,1]**2]          # (100,3)

# optional Tikhonov (ridge) regularisation
gamma = 0.0                                            # 0  means no reg.
A = Q.T @ Q + gamma*np.eye(3)                           # (3×3)
B = Q.T @ R                                             # (3×3)
W = np.linalg.solve(A, B).T                             # (3×3)  columns = v̂k

# ========== 4.  full quadratic reconstruction ==============================
S_quad = s_ref + S_lin + Q @ W.T                        # Γ(z)

# ========== 5.  sample manifold surface for plotting ========================
z1_min, z1_max = z[:,0].min() - 0.1, z[:,0].max() + 0.1
z2_min, z2_max = z[:,1].min() - 0.1, z[:,1].max() + 0.1
g1 = np.linspace(z1_min, z1_max, 35)
g2 = np.linspace(z2_min, z2_max, 35)
G1, G2 = np.meshgrid(g1, g2)
Zgrid  = np.vstack((G1.ravel(), G2.ravel()))            # (2,N)

# linear part on the grid
Surf_lin  = V @ Zgrid                                   # (3,N)

# quadratic correction on the grid
Qgrid = np.vstack((Zgrid[0]**2,
                   Zgrid[0]*Zgrid[1],
                   Zgrid[1]**2))                        # (3,N)
Surf_full = s_ref[:,None] + Surf_lin + W @ Qgrid        # (3,N)
X, Y, Z = [a.reshape(G1.shape) for a in Surf_full]

# ========== 6.  error report (optional) =====================================
err_lin  = np.linalg.norm(S - (s_ref + S_lin),  2)        # total 2-norm
err_quad = np.linalg.norm(S - S_quad,          2)
print(f"‖S–S_lin‖ = {err_lin:9.6f}   ‖S–S_quad‖ = {err_quad:9.6f}")

# ========== 7.  visualisation ==============================================
fig = plt.figure(figsize=(9,7))
ax  = fig.add_subplot(111, projection='3d')

ax.plot3D(*S.T,       'ko', ms=4, label=r'\textit{trajectory} $\mathbf{s}(t)$')
ax.plot3D(*S_quad.T,  color='magenta', lw=2,
          label=r'\textit{approximated trajectory}')
ax.plot_surface(X, Y, Z, color='plum', alpha=0.15, edgecolor='gray', lw=0.3)

ax.set(xlabel=r'$s_1$', ylabel=r'$s_2$', zlabel=r'$s_3$',
       xlim=(-1.5,1.5), ylim=(-1.5,1.5), zlim=(-0.6,0.6))
ax.set_title(r'\textbf{Quadratic manifold}', pad=20)
ax.view_init(elev=15, azim=225)

ax.legend(loc='upper right', handles=[
    Line2D([],[], color='k', marker='o', ls='None',
           label=r'trajectory $\mathbf{s}(t)$'),
    Line2D([],[], color='magenta', lw=2,
           label=r'approximated trajectory'),
    Patch(facecolor='plum', alpha=0.3, edgecolor='gray',
          label=r'quadratic manifold')
])

plt.tight_layout()
plt.savefig("quadratic_manifold.pdf", bbox_inches='tight')
plt.show()
