# rbf_staged.py
# ---------------------------------------------------------------
# Radial Basis Function (RBF) interpolation — staged visualization
# Saves:
#   rbf_stage1_true.png       (true function only)
#   rbf_stage2_true_train.png (true + training points)
#   rbf_stage3_rbf.png        (true + points + RBF curve)
#   rbf_stages_1x3.png        (composite 1x3)

import numpy as np
import matplotlib.pyplot as plt
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold

# ---------- Matplotlib style (no LaTeX requirement) ----------
plt.rcParams.update({
    "text.usetex": False,      # keep False to avoid LaTeX dependency
    "font.family": "serif",
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})

# ---------- Ground-truth function (smooth w/ varying curvature) ----------
def f_true(x):
    return np.sin(2*np.pi*x) + 0.3*np.cos(5*np.pi*x)

# Deliberately uneven training locations (gap to make interpolation evident)
X_points = [0.02, 0.23, 0.40, 0.74, 0.81, 0.87]
X = np.array(X_points, dtype=float).reshape(-1, 1)
y = f_true(X).ravel()

# Plot grid
xx = np.linspace(0, 1, 600).reshape(-1, 1)
f_grid = f_true(xx).ravel()

# ---------- Fit RBF via Kernel Ridge Regression (Gaussian kernel) ----------
# We choose a very small alpha to emulate interpolation, and select gamma by CV.
param_grid = {
    "alpha": [1e-10, 3e-10, 1e-9],                  # tiny ridge for stability
    "gamma": 10.0 ** np.linspace(0.0, 2.5, 13),     # kernel width (= 1/(2*eps^2))
}
krr = KernelRidge(kernel="rbf")
cv = KFold(n_splits=5, shuffle=True, random_state=7)
grid = GridSearchCV(krr, param_grid=param_grid, cv=cv,
                    scoring="neg_mean_squared_error", n_jobs=-1)
grid.fit(X, y)
rbf_model = grid.best_estimator_
rbf_pred = rbf_model.predict(xx)

print("\n=== RBF (deterministic via KRR + CV) ===")
print("Best params:", grid.best_params_)
rmse_train = np.sqrt(np.mean((rbf_model.predict(X) - y)**2))
print("Training RMSE:", rmse_train)

# ---------- Plot helpers ----------
def base_axes(figsize=(5.2, 4.0), title=""):
    fig, ax = plt.subplots(figsize=figsize)
    ax.set_xlabel(r"$x$")
    ax.set_ylabel(r"$y$")
    ax.set_title(title)
    ax.plot(xx, f_grid, "--", lw=1.3, color="0.55", label=r"true $f(x)$")
    return fig, ax

def scatter_training(ax):
    ax.scatter(X, y, marker="x", c="k", s=50, linewidths=2, label="training")

# ---------- Stage 1: true function only ----------
fig1, ax1 = base_axes(title="Stage 1: ground truth")
ax1.legend(loc="best", frameon=True)
fig1.tight_layout()
fig1.savefig("rbf_stage1_true.png", dpi=220, bbox_inches="tight")

# ---------- Stage 2: + training points ----------
fig2, ax2 = base_axes(title="Stage 2: add training points")
scatter_training(ax2)
ax2.legend(loc="best", frameon=True)
fig2.tight_layout()
fig2.savefig("rbf_stage2_true_train.png", dpi=220, bbox_inches="tight")

# ---------- Stage 3: + RBF interpolation ----------
fig3, ax3 = base_axes(title="Stage 3: RBF interpolation")
scatter_training(ax3)
ax3.plot(xx, rbf_pred, "-", lw=2.0, color="#C02030", label="RBF interpolation")
ax3.legend(loc="best", frameon=True)
fig3.tight_layout()
fig3.savefig("rbf_stage3_rbf.png", dpi=220, bbox_inches="tight")

# ---------- Composite 1x3 ----------
figC, axes = plt.subplots(1, 3, figsize=(13.5, 3.9), sharey=True)
# Panel 1
axes[0].plot(xx, f_grid, "--", lw=1.3, color="0.55", label=r"true $f(x)$")
axes[0].set_title("1) Ground truth"); axes[0].set_xlabel(r"$x$"); axes[0].set_ylabel(r"$y$")
# Panel 2
axes[1].plot(xx, f_grid, "--", lw=1.3, color="0.55")
axes[1].scatter(X, y, marker="x", c="k", s=50, linewidths=2, label="training")
axes[1].set_title("2) + training points"); axes[1].set_xlabel(r"$x$")
# Panel 3
axes[2].plot(xx, f_grid, "--", lw=1.3, color="0.55")
axes[2].scatter(X, y, marker="x", c="k", s=50, linewidths=2)
axes[2].plot(xx, rbf_pred, "-", lw=2.0, color="#C02030", label="RBF interpolation")
axes[2].set_title("3) + RBF interpolation"); axes[2].set_xlabel(r"$x$")
for ax in axes: ax.grid(True); ax.legend(loc="lower left", frameon=True)
plt.tight_layout()
figC.savefig("rbf_stages_1x3.png", dpi=220, bbox_inches="tight")

print("\nSaved: rbf_stage1_true.png, rbf_stage2_true_train.png, rbf_stage3_rbf.png, rbf_stages_1x3.png")
