#!/usr/bin/env python3
"""
piecewise_pod_manifold.py
-------------------------

Builds N local linear POD manifolds on a closed 3-D trajectory, blends the
local reconstructions in overlaps, and plots a SINGLE 3-D figure.

Call `plot_piecewise_manifold(S, n_seg=3, overlap=0.25)` with

    S        – (m,3) ndarray containing the trajectory
    n_seg    – how many evenly-spaced circular segments you want
    overlap  – fractional overlap between neighbouring windows
               (0 → disjoint, 0.99 → almost identical windows)

The function returns the blended reconstruction and the figure handle, so
you can re-use both if needed.
"""

from __future__ import annotations
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from itertools import cycle

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
    fname: str | None = "piecewise_linear_manifold.pdf"
):
    """Main driver; see doc-string at top."""
    m = len(S)
    seg_len = int(np.ceil((1 + overlap) * m / n_seg))        # window size
    step    = int(np.floor((1 - overlap) * seg_len))         # step size

    if colours is None:
        colours = ['skyblue', 'palegreen', 'plum', 'lightcoral',
                   'gold', 'lightsalmon', 'turquoise']
    col_cycle = cycle(colours)

    local_data = []          # store per-segment information
    blend_sum  = np.zeros_like(S, dtype=float)
    blend_cnt  = np.zeros((m, 1), dtype=float)

    # -----------------------------------------------------------------
    # 1. Loop over 'centre' windows  (fixes wrap-around discontinuity)
    # -----------------------------------------------------------------
    length = int(np.ceil((1 + overlap) * m / n_seg))   # window length
    half   = length // 2
    centres = np.linspace(0, m, n_seg, endpoint=False, dtype=int)

    for k, centre in enumerate(centres):
        seg, idx = circular_slice(S, centre - half, length)
        recon, V, sref = local_rank2_pod(seg)

        colour = next(col_cycle)
        local_data.append(dict(idx=idx, recon=recon, V=V,
                            z=(seg - sref) @ V, ref=sref, colour=colour))

        blend_sum[idx] += recon
        blend_cnt[idx] += 1


    #blend_cnt[blend_cnt == 0] = 1          # prevent divide-by-zero
    S_blend = blend_sum / blend_cnt

    # -----------------------------------------------------------------
    # 2. Plot – one 3-D panel
    # -----------------------------------------------------------------
    fig = plt.figure(figsize=(9, 7))
    ax  = fig.add_subplot(111, projection='3d')

    # original trajectory
    ax.plot3D(*S.T, 'ko', ms=4, label=r'\textit{trajectory} $\mathbf{s}(t)$')

    # local curves + planes
    grid = np.linspace(-1.2, 1.2, 20)
    for rec in local_data:
        col = rec['colour']
        # dashed local reconstruction
        ax.plot3D(*rec['recon'].T, ls='--', lw=2, color='darkorange')

        # tiny plane for visual cue
        z_mean = rec['z'].mean(axis=0)
        A, B   = np.meshgrid(grid + z_mean[0], grid + z_mean[1])
        Zred   = np.vstack((A.ravel(), B.ravel()))
        plane  = rec['ref'][:, None] + rec['V'] @ Zred
        Xp, Yp, Zp = [a.reshape(A.shape) for a in plane]
        ax.plot_surface(Xp, Yp, Zp, color=col, alpha=0.15,
                        edgecolor='gray', linewidth=0.3)

    # blended curve
    ax.plot3D(*S_blend.T, lw=3, color='crimson',
              label=r'\textit{approximated trajectory}')

    # axes / view
    ax.set(xlabel=r'$s_1$', ylabel=r'$s_2$', zlabel=r'$s_3$',
           xlim=(-1.5, 1.5), ylim=(-1.5, 1.5), zlim=(-0.6, 0.6))
    ax.set_title(r'\textbf{Piece–wise linear manifolds}', pad=20)
    ax.view_init(elev=elev, azim=azim)

    # legend (dynamic depending on n_seg)
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
    ax.legend(handles=legend_entries, loc='upper right')

    plt.tight_layout()
    if fname:
        fig.savefig(fname, bbox_inches='tight')
        print("saved →", fname)
    return S_blend, fig


# ---------------------------------------------------------------------
# 3. QUICK DEMO  (remove or guard with __name__ == '__main__' if needed)
# ---------------------------------------------------------------------
if __name__ == "__main__":
    # build the same toy trajectory used in all your previous plots
    t_demo = np.linspace(0, 2*np.pi, 100)
    S_demo = np.c_[np.cos(t_demo),
                   np.sin(t_demo),
                   0.5*np.cos(2*t_demo)]

    # try 2, 3 or 5 segments – overlap = 0.3 looks nice
    plot_piecewise_manifold(S_demo, n_seg=4, overlap=0.1,
                            fname="piecewise_linear_manifold.pdf")
    plt.show()

