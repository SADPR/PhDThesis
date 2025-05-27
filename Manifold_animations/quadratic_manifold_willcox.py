import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

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

# --- 1. data -------------------------------------------------
t = np.linspace(0, 2*np.pi, 100)
S = np.c_[np.cos(t), np.sin(t), 0.5*np.cos(2*t)]          # (100,3)

# choose reference – mean works best here
s_ref = S[0]#S.mean(axis=0)            # try also S[0]
S_shift = S - s_ref

# --- 2. POD --------------------------------------------------
U, s, VT = np.linalg.svd(S_shift, full_matrices=False)
V = VT[:2].T                      # (3×2)

# --- 3. orthogonal projection & residual --------------------
hat_s = S_shift @ V               # (100×2)
S_lin = hat_s @ V.T               # linear part
# force residual strictly into V_perp
E = S_shift - S_lin               # same as (I-VVᵀ)S_shift numerically

# --- 4. quadratic features ----------------------------------
Q = np.c_[hat_s[:,0]**2, hat_s[:,0]*hat_s[:,1], hat_s[:,1]**2]  # (100×3)

# ridge regularisation (γ·I)  -> W solves (QᵀQ + γI)Wᵀ = QᵀE
γ = 0.0
A = Q.T @ Q + γ*np.eye(3)
B = Q.T @ E
W = np.linalg.solve(A, B).T       # (3×3)

# --- 5. reconstruction --------------------------------------
S_quad = s_ref + S_lin + Q @ W.T

# --- 6. build surface for visualisation ---------------------
α = np.linspace(hat_s[:,0].min()-0.1, hat_s[:,0].max()+0.1, 35)
β = np.linspace(hat_s[:,1].min()-0.1, hat_s[:,1].max()+0.1, 35)
A,B = np.meshgrid(α,β)
AB  = np.vstack((A.ravel(), B.ravel()))
V_part = V @ AB
Q_grid = np.vstack([AB[0]**2, AB[0]*AB[1], AB[1]**2])
surf   = s_ref[:,None] + V_part + W @ Q_grid
X,Y,Z  = [a.reshape(A.shape) for a in surf]

# --- 7. plot -------------------------------------------------
fig = plt.figure(figsize=(9,7))
ax  = fig.add_subplot(111, projection='3d')
ax.plot3D(*S.T, 'ko', ms=4, label=r'\textit{trajectory} $\mathbf{s}(t)$')
ax.plot3D(*S_quad.T, color='magenta', lw=2, label=r'\textit{approximated trajectory}')
ax.plot_surface(X,Y,Z, color='plum', alpha=0.15, edgecolor='gray', linewidth=0.3)

ax.set(xlim=(-1.5,1.5), ylim=(-1.5,1.5), zlim=(-0.6,0.6),
       xlabel=r'$s_1$', ylabel=r'$s_2$', zlabel=r'$s_3$')
ax.set_title(r'\textbf{Quadratic manifold}', pad=20)
ax.view_init(elev=15, azim=225)

ax.legend(loc='upper right', handles=[
    Line2D([],[], color='k', marker='o', ls='None', label=r'trajectory $\mathbf{s}(t)$'),
    Line2D([],[], color='magenta', lw=2,          label=r'approximated trajectory'),
    Patch(facecolor='plum', alpha=0.3, edgecolor='gray', label=r'quadratic manifold')
])

plt.tight_layout()
plt.savefig("quadratic_manifold_paper_style.pdf", bbox_inches='tight')
plt.show()
