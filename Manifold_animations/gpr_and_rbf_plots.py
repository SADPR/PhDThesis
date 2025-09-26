# gpr_vs_rbf_clean.py
# ---------------------------------------------------------------
# GPR (noise-free, log-marginal likelihood) vs RBF (deterministic, CV)
# Shows larger GPR uncertainty where data are sparse (big gap on purpose).
# Saves: gpr_rbf_side_by_side.png/png, gpr_only.png, rbf_only.png

import numpy as np
import matplotlib.pyplot as plt
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel
from sklearn.kernel_ridge import KernelRidge
from sklearn.model_selection import GridSearchCV, KFold

# ----- Matplotlib style (no LaTeX dependency) -----
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.titlesize": 15,
    "axes.labelsize": 12,
    "legend.fontsize": 11,
    "axes.grid": True,
    "grid.color": "0.85",
    "grid.linestyle": "--",
    "grid.linewidth": 0.6,
})

# ----- Ground-truth function (same look as your slide) -----
def f_true(x):
    return np.sin(2*np.pi*x) + 0.3*np.cos(5*np.pi*x)

# ----- Training inputs: deliberately clustered with a wide gap -----
X_points = [
    0.02, 0.23, 0.40,     
    0.74, 0.81, 0.87#, 0.96                               # few points right
]
X = np.array(X_points, dtype=float).reshape(-1, 1)

# Noise-free targets (interpolation setting)
y = f_true(X).ravel()

# Dense plot grid
xx = np.linspace(0, 1, 600).reshape(-1, 1)
f_grid = f_true(xx).ravel()

# =========================
# 1) GPR (noise-free)
# =========================
# Kernel: constant * RBF; no WhiteKernel (interpolating GP)
kernel = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(length_scale=0.2, length_scale_bounds=(1e-2, 5.0))
gpr = GaussianProcessRegressor(kernel=kernel, normalize_y=True,
                               random_state=7, n_restarts_optimizer=5)
gpr.fit(X, y)
gpr_mean, gpr_std = gpr.predict(xx, return_std=True)

print("\n=== GPR (noise-free) ===")
print("Optimized kernel:", gpr.kernel_)
print("Training RMSE (should be ~0 for interpolation):",
      np.sqrt(np.mean((gpr.predict(X) - y)**2)))

# =========================
# 2) RBF (deterministic) via Kernel Ridge + CV
# =========================
# KernelRidge with RBF kernel; choose alpha (ridge) and gamma by CV.
krr = KernelRidge(kernel='rbf')
param_grid = {
    "alpha": 10.0 ** np.arange(-10, -4),         # very small ridge to mimic interpolation
    "gamma": 10.0 ** np.linspace(0.0, 2.5, 13),  # RBF width
}
cv = KFold(n_splits=5, shuffle=True, random_state=7)
grid = GridSearchCV(krr, param_grid=param_grid, cv=cv,
                    scoring='neg_mean_squared_error', n_jobs=-1)
grid.fit(X, y)
krr_best = grid.best_estimator_
rbf_pred = krr_best.predict(xx)

print("\n=== RBF (deterministic, CV) ===")
print("Best params:", grid.best_params_)
print("Training RMSE:", np.sqrt(np.mean((krr_best.predict(X) - y)**2)))

# Compare the two means
max_abs_diff = np.max(np.abs(gpr_mean - rbf_pred))
print("\nMax |GPR_mean - RBF_pred| on grid:", max_abs_diff)

# =========================
# Plotting
# =========================
def scatter_training(ax):
    ax.scatter(X, y, marker='x', c='k', s=50, linewidths=2, label='training')

fig, axes = plt.subplots(1, 2, figsize=(11.5, 4), sharey=True)

# Left: GPR
ax = axes[0]
ax.plot(xx, f_grid, '--', lw=1.2, color='0.55', label=r'true $f(x)$')
ax.fill_between(xx.ravel(), gpr_mean - 2*gpr_std, gpr_mean + 2*gpr_std,
                color='#E9CFA5', alpha=0.85, edgecolor='none', label=r'95$\%$ confidence interval')
ax.plot(xx, gpr_mean, '-', color='#C02030', lw=2.0, label='GPR mean')
scatter_training(ax)
ax.set_title(r'GPR (probabilistic)')
ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')
ax.legend(loc='lower left', frameon=True)

# Right: RBF
ax = axes[1]
ax.plot(xx, f_grid, '--', lw=1.2, color='0.55', label=r'true $f(x)$')
ax.plot(xx, rbf_pred, '-', color='blue', lw=2.0, label='RBF interpolation')
scatter_training(ax)
ax.set_title(r'RBF (deterministic)')
ax.set_xlabel(r'$x$')
ax.legend(loc='lower left', frameon=True)

plt.tight_layout()
fig.savefig('gpr_rbf_side_by_side.png', dpi=220, bbox_inches='tight')
plt.show()  

# Single-panel exports
# GPR only
fig_gpr, ax = plt.subplots(figsize=(5.8, 4.2))
ax.plot(xx, f_grid, '--', lw=1.2, color='0.55', label='true $f(x)$')
ax.fill_between(xx.ravel(), gpr_mean - 2*gpr_std, gpr_mean + 2*gpr_std,
                color='#E9CFA5', alpha=0.85, edgecolor='none', label=r'95$\%$ confidence interval')
ax.plot(xx, gpr_mean, '-', color='#C02030', lw=2.2, label='GPR mean')
scatter_training(ax)
ax.set_title(r'GPR (probabilistic)')
ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')
ax.legend(loc='best', frameon=True)
plt.tight_layout()
fig_gpr.savefig('gpr_only.png', bbox_inches='tight')

# RBF only
fig_rbf, ax = plt.subplots(figsize=(5.8, 4.2))
ax.plot(xx, f_grid, '--', lw=1.2, color='0.55', label='true $f(x)$')
ax.plot(xx, rbf_pred, '-', color='#C02030', lw=2.2, label='RBF interpolation')
scatter_training(ax)
ax.set_title(r'RBF (deterministic)')
ax.set_xlabel(r'$x$'); ax.set_ylabel(r'$y$')
ax.legend(loc='best', frameon=True)
plt.tight_layout()
fig_rbf.savefig('rbf_only.png', bbox_inches='tight')

print("\nSaved: gpr_rbf_side_by_side.[png|png], gpr_only.png, rbf_only.png")
