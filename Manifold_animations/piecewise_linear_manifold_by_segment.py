#!/usr/bin/env python3
"""
piecewise_pod_manifold.py
-------------------------

Builds N local linear POD manifolds on a closed 3-D trajectory, blends the
local reconstructions in overlaps, and plots BOTH:
  - individual plots for each local manifold
  - a final combined piecewise-linear manifold plot.

Run with:

    python piecewise_pod_manifold.py

or import and call `plot_piecewise_manifold(S, n_seg, overlap)`.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from itertools import cycle
import os

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

# ---------------------------------------------------------------------
def circular_slice(a: np.ndarray, start: int, length: int) -> tuple[np.ndarray, np.ndarray]:
    """Return view of a circular slice and the corresponding global indices."""
    idx = np.arange(start, start + length) % len(a)
    return a[idx], idx


# ---------------------------------------------------------------------
def local_rank2_pod(segment: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Rank-2 POD basis and reconstruction for ONE segment."""
    s_ref = segment[0]
    shift = segment - s_ref
    _, _, vt = np.linalg.svd(shift, full_matrices=False)
    V      = vt[:2].T                                    # (3×2)
    z      = shift @ V                                   # reduced coords
    recon  = s_ref + z @ V.T
    return recon, V, s_ref


# ---------------------------------------------------------------------
def plot_piecewise_manifold(
    S: np.ndarray,
    n_seg: int       = 3,
    overlap: float   = 0.25,
    colours: list[str] | None = None,
    elev: int = 15,
    azim: int = 225,
    out_dir: str = "piecewise_outputs"
):
    """Main driver; see doc-string at top."""
    m = len(S)
    seg_len = int(np.ceil((1 + overlap) * m / n_seg))        # window size

    if colours is None:
        colours = ['skyblue', 'palegreen', 'plum', 'lightcoral',
                   'gold', 'lightsalmon', 'turquoise']
    col_cycle = cycle(colours)

    local_data = []          # store per-segment information
    blend_sum  = np.zeros_like(S, dtype=float)
    blend_cnt  = np.zeros((m, 1), dtype=float)

    length = seg_len
    half   = length // 2
    centres = np.linspace(0, m, n_seg, endpoint=False, dtype=int)

    # Create output directory
    os.makedirs(out_dir, exist_ok=True)

    grid = np.linspace(-1.2, 1.2, 20)

    # -----------------------------------------------------------------
    # 1. Loop over 'centre' windows: build local PODs + save individual plots
    # -----------------------------------------------------------------
    for idx, centre in enumerate(centres, start=1):
        seg, idxs = circular_slice(S, centre - half, length)
        recon, V, sref = local_rank2_pod(seg)

        colour = next(col_cycle)
        local_data.append(dict(idx=idxs, recon=recon, V=V,
                               z=(seg - sref) @ V, ref=sref, colour=colour))

        blend_sum[idxs] += recon
        blend_cnt[idxs] += 1

        # ---- Individual local manifold plot ----
        fig_local = plt.figure(figsize=(9, 7))
        ax_local  = fig_local.add_subplot(111, projection='3d')

        ax_local.plot3D(*S.T, 'ko', ms=4, label=r'\textit{trajectory} $\mathbf{s}(t)$')
        ax_local.plot3D(*recon.T, ls='--', lw=2, color='darkorange',
                        label=r'local reconstruction')

        z_mean = np.mean((seg - sref) @ V, axis=0)
        A, B   = np.meshgrid(grid + z_mean[0], grid + z_mean[1])
        Zred   = np.vstack((A.ravel(), B.ravel()))
        plane  = sref[:, None] + V @ Zred
        Xp, Yp, Zp = [a.reshape(A.shape) for a in plane]
        ax_local.plot_surface(Xp, Yp, Zp, color=colour, alpha=0.2,
                              edgecolor='gray', linewidth=0.3)

        ax_local.set(xlabel=r'$s_1$', ylabel=r'$s_2$', zlabel=r'$s_3$',
                     xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-0.6, 0.6))
        ax_local.set_title(rf'\textbf{{Local manifold {idx}}}', pad=20)
        ax_local.view_init(elev=elev, azim=azim)
        ax_local.legend(loc='upper right')

        local_fname = os.path.join(out_dir, f"local_manifold_{idx}.pdf")
        fig_local.savefig(local_fname, bbox_inches='tight')
        plt.close(fig_local)
        print(f"saved → {local_fname}")

    S_blend = blend_sum / blend_cnt

    # -----------------------------------------------------------------
    # 2. Final combined piecewise manifold plot
    # -----------------------------------------------------------------
    fig_comb = plt.figure(figsize=(9, 7))
    ax_comb  = fig_comb.add_subplot(111, projection='3d')

    ax_comb.plot3D(*S.T, 'ko', ms=4, label=r'\textit{trajectory} $\mathbf{s}(t)$')

    for rec in local_data:
        col = rec['colour']
        ax_comb.plot3D(*rec['recon'].T, ls='--', lw=2, color='darkorange')

        z_mean = rec['z'].mean(axis=0)
        A, B   = np.meshgrid(grid + z_mean[0], grid + z_mean[1])
        Zred   = np.vstack((A.ravel(), B.ravel()))
        plane  = rec['ref'][:, None] + rec['V'] @ Zred
        Xp, Yp, Zp = [a.reshape(A.shape) for a in plane]
        ax_comb.plot_surface(Xp, Yp, Zp, color=col, alpha=0.15,
                             edgecolor='gray', linewidth=0.3)

    ax_comb.plot3D(*S_blend.T, lw=3, color='crimson',
                   label=r'\textit{approximated trajectory}')

    ax_comb.set(xlabel=r'$s_1$', ylabel=r'$s_2$', zlabel=r'$s_3$',
                xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-0.6, 0.6))
    ax_comb.set_title(r'\textbf{Piece–wise linear manifolds (combined)}', pad=20)
    ax_comb.view_init(elev=elev, azim=azim)

    legend_entries = [
        Line2D([], [], marker='o', ls='None', color='k',
               label=r'trajectory $\mathbf{s}(t)$'),
        Line2D([], [], ls='--', lw=2, color='darkorange',
               label=r'local reconstructions'),
        Line2D([], [], lw=3, color='crimson',
               label=r'approximated trajectory')
    ]
    for i, rec in enumerate(local_data, start=1):
        legend_entries.append(
            Patch(facecolor=rec['colour'], alpha=0.3, edgecolor='gray',
                  label=rf'linear manifold {i}')
        )
    ax_comb.legend(handles=legend_entries, loc='upper right')

    fig_comb.tight_layout()
    final_fname = os.path.join(out_dir, "piecewise_linear_combined.pdf")
    fig_comb.savefig(final_fname, bbox_inches='tight')
    plt.close(fig_comb)
    print(f"saved → {final_fname}")

    return S_blend


# ---------------------------------------------------------------------
# 3. QUICK DEMO  (remove or guard with __name__ == '__main__' if needed)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    t_demo = np.linspace(0, 2*np.pi, 100)
    S_demo = np.c_[np.cos(t_demo),
                   np.sin(t_demo),
                   0.5*np.cos(2*t_demo)]

    plot_piecewise_manifold(S_demo, n_seg=4, overlap=0.1)


