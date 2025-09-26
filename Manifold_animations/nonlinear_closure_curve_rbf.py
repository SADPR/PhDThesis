# rbf_closure_1d_view.py
# -----------------------------------------------------
# Simplified closure view:
# Plot q (latent coordinate) vs q_bar (closure),
# with RBF interpolation curve.

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import Rbf

# ---- Style ----
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.titlesize": 18,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12
})

# === 1) Example trajectory in R^3 ===
t = np.linspace(0, 2*np.pi, 120, endpoint=False)
S = np.c_[np.cos(t), np.sin(t), 0.5*np.cos(2*t)].astype(float)

u_ref = S[0]
S_shift = S - u_ref

# === 2) POD basis: n=2 primaries, n_bar=1 closure ===
U, svals, VT = np.linalg.svd(S_shift, full_matrices=False)
V = VT.T
V_p = V[:, :2]
V_s = V[:,  2:3]

# === 3) Reduced coordinates ===
q_p   = S_shift @ V_p          # (N,2)
q_s   = (S_shift @ V_s).ravel()# (N,)

# For symmetry, just take q1 as "q"
q = q_p[:,0]

# === 4) Fit RBF (q -> q_bar) ===
rbf = Rbf(q, q_s, function='multiquadric', smooth=1e-8)
q_line = np.linspace(q.min(), q.max(), 400)
qbar_pred = rbf(q_line)

# === 5) Plot ===
fig, ax = plt.subplots(figsize=(7,5))
ax.scatter(q, q_s, c='k', s=22, label=r'training $(q,\bar q)$')
ax.plot(q_line, qbar_pred, 'r-', lw=2, label=r'RBF interpolation')

ax.set_xlabel(r'$q$')
ax.set_ylabel(r'$\bar q$')
ax.set_title(r'\textbf{Closure map: } $\bar q = \mathcal{N}_{\mathrm{RBF}}(q)$')
ax.legend(frameon=True, loc='best')

plt.tight_layout()
plt.savefig("rbf_closure_q_vs_qbar.pdf", format="pdf", bbox_inches='tight')
plt.show()



