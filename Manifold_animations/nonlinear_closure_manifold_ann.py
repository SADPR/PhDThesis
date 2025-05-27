import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
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
S = np.c_[np.cos(t), np.sin(t), 0.5*np.cos(2*t)].astype(np.float32)
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

# === 4. Define ANN closure model ============================================
class ClosureANN(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(2, 16),
            nn.ELU(),
            nn.Linear(16, 16),
            nn.ELU(),
            nn.Linear(16, 1)
        )
    def forward(self, x):
        return self.net(x)

model = ClosureANN()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

# === 5. Train ANN model with progress output ================================
X_train = torch.from_numpy(q_p)
y_train = torch.from_numpy(q_s[:, None])

for epoch in range(1000):
    model.train()
    optimizer.zero_grad()
    loss = criterion(model(X_train), y_train)
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1:4d} | Loss: {loss.item():.6e}")

# === 6. Predict and reconstruct =============================================
with torch.no_grad():
    q_s_pred = model(X_train).numpy().ravel()
S_pred = u_ref + q_p @ V_p.T + np.outer(q_s_pred, V_s.ravel())

# === 7. Surface from latent space ===========================================
z1 = np.linspace(q_p[:, 0].min() - 0.2, q_p[:, 0].max() + 0.2, 60)
z2 = np.linspace(q_p[:, 1].min() - 0.2, q_p[:, 1].max() + 0.2, 60)
Z1, Z2 = np.meshgrid(z1, z2)
grid_qp = np.stack([Z1.ravel(), Z2.ravel()], axis=1)

with torch.no_grad():
    q_s_grid = model(torch.from_numpy(grid_qp).float()).numpy().ravel()

# === 8. Map back to 3D using decoder logic ==================================
V_p_term = (V_p @ grid_qp.T).reshape(3, 60, 60)
V_s_term = (V_s * q_s_grid.reshape(1, -1)).reshape(3, 60, 60)
u_ref_tensor = u_ref.reshape(3, 1, 1)
U_grid = u_ref_tensor + V_p_term + V_s_term
X_surf, Y_surf, Z_surf = U_grid

# === 9. Plot ================================================================
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')
ax.plot(S[:, 0], S[:, 1], S[:, 2], 'ko', ms=4, label='trajectory')
ax.plot(S_pred[:, 0], S_pred[:, 1], S_pred[:, 2], '--', color='mediumvioletred', lw=2, label='ANN-based reconstruction')
ax.plot_surface(X_surf, Y_surf, Z_surf, alpha=0.25, color='orchid', edgecolor='gray', linewidth=0.3)

ax.set(xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-0.6, 0.6),
       xlabel=r'$s_1$', ylabel=r'$s_2$', zlabel=r'$s_3$')
ax.set_title(r'\textbf{Nonlinear closure manifold (ANN-based)}', pad=18)
ax.view_init(elev=15, azim=225)

legend_elements = [
    Line2D([], [], color='k', marker='o', linestyle='None', label=r'trajectory $\mathbf{s}(t)$'),
    Line2D([], [], color='mediumvioletred', linestyle='--', lw=2, label=r'approximated trajectory'),
    Patch(facecolor='orchid', edgecolor='gray', label=r'nonlinear closure manifold', alpha=0.3)
]
ax.legend(handles=legend_elements, loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig("nonlinear_closure_manifold_ann.pdf", format="pdf", bbox_inches='tight')
plt.show()
