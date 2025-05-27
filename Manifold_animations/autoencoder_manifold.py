# autoencoder_manifold.py  –– “Autoencoder-based nonlinear manifold” figure
# --------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines  import Line2D
from matplotlib.patches import Patch
import torch
import torch.nn as nn
import torch.optim as optim

# ---------- 0. Global LaTeX / font style -----------------------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family" : "serif",
    "axes.titlesize": 20,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14,
})

# ---------- 1. Trajectory  s(t)  -------------------------------------------
t = np.linspace(0, 2*np.pi, 100)
S = np.c_[np.cos(t), np.sin(t), 0.5*np.cos(2*t)].astype(np.float32)   # (100,3)

# ---------- 2. Two–dimensional auto-encoder --------------------------------
class AE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(3, 16), nn.ELU(),
            nn.Linear(16, 2)
        )
        self.decoder = nn.Sequential(
            nn.Linear(2, 16), nn.ELU(),
            nn.Linear(16, 3)
        )
    def forward(self, x):                        # full AE pass
        return self.decoder(self.encoder(x))

model     = AE()
criterion = nn.MSELoss()
optimiser = optim.Adam(model.parameters(), lr=1e-2)

# ---------- 3. Training -----------------------------------------------------
x_tensor = torch.from_numpy(S)
loader   = torch.utils.data.DataLoader(
              torch.utils.data.TensorDataset(x_tensor),
              batch_size=16, shuffle=True)

for epoch in range(800):                        # ~½ s on CPU
    for (batch,) in loader:
        optimiser.zero_grad()
        loss = criterion(model(batch), batch)
        loss.backward()
        optimiser.step()
    if (epoch+1) % 100 == 0:
        print(f"epoch {epoch+1:>3d}  |  loss ≈ {loss.item():.3e}")

# ---------- 4. Inference -----------------------------------------------------
model.eval()
with torch.no_grad():
    X_recon   = model(x_tensor).numpy()               # (100,3)
    Z_latent  = model.encoder(x_tensor).numpy()       # (100,2)

# ---------- 5. Decode a latent grid to draw the manifold surface ------------
pad   = 0.25                                          # visual margin
z1    = np.linspace(Z_latent[:,0].min()-pad, Z_latent[:,0].max()+pad, 35)
z2    = np.linspace(Z_latent[:,1].min()-pad, Z_latent[:,1].max()+pad, 35)
Z1, Z2 = np.meshgrid(z1, z2)
latent_grid = torch.from_numpy(
                 np.stack([Z1.ravel(), Z2.ravel()], axis=1)
             ).float()
with torch.no_grad():
    decoded = model.decoder(latent_grid).numpy()      # (35²,3)

X_surf = decoded[:,0].reshape(Z1.shape)
Y_surf = decoded[:,1].reshape(Z1.shape)
Z_surf = decoded[:,2].reshape(Z1.shape)

# ---------- 6. Plot ----------------------------------------------------------
fig = plt.figure(figsize=(9,7))
ax  = fig.add_subplot(111, projection='3d')

# 6.1 curves
ax.plot3D(*S.T       , 'ko',  ms=4,
          label=r'\textit{trajectory} $\mathbf{s}(t)$')
ax.plot3D(*X_recon.T , color='orange', lw=2, ls='--',
          label=r'\textit{approximated trajectory}')

# 6.2 manifold surface
ax.plot_surface(X_surf, Y_surf, Z_surf,
                color='royalblue', alpha=0.15,
                edgecolor='gray', linewidth=0.25)

# 6.3 axes / legend
ax.set(xlim=(-1.5,1.5), ylim=(-1.5,1.5), zlim=(-0.6,0.6),
       xlabel=r'$s_1$', ylabel=r'$s_2$', zlabel=r'$s_3$')
ax.set_title(r'\textbf{Autoencoder manifold}', pad=20)
ax.view_init(elev=15, azim=225)

ax.legend(handles=[
    Line2D([], [], color='k',       marker='o', ls='None',
           label=r'trajectory $\mathbf{s}(t)$'),
    Line2D([], [], color='orange',  lw=2,      ls='--',
           label=r'approximated trajectory'),
    Patch  (facecolor='royalblue',  edgecolor='gray', alpha=0.3,
            label=r'autoencoder manifold')
], loc='upper right', frameon=True)

plt.tight_layout()
plt.savefig("autoencoder_manifold.pdf", bbox_inches='tight')
plt.show()
