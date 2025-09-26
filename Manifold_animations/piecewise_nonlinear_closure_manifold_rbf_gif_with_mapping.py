#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Two local nonlinear manifolds (𝒩_{Φ,1} and 𝒩_{Φ,2}) with RBF closures.
Animation: trajectory -> (manifold 1 in/out) -> (manifold 2 in/out) ->
          re-add both manifolds -> draw blended nonlinear trajectory -> spin.
"""

import numpy as np
from scipy.interpolate import Rbf
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

# ================== STYLE ====================
plt.rcParams.update({
    "text.usetex": True,         # set to False if LaTeX is unavailable
    "font.family": "serif",
    "axes.titlesize": 20,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

# ================== CONFIG ===================
GIF_NAME          = "two_nonlinear_manifolds.gif"
FPS               = 24
SURF_ALPHA        = 0.18
GRID_ALPHA        = 0.35
SURF_FADE_FRAMES  = 20
LOCAL_DRAW_FRAMES = 60
LOCAL_FADE_FRAMES = 12
HOLD_A_FRAMES     = 10
FINAL_SURF_FADE   = 18
BLEND_DRAW_FRAMES = 100
HOLD_B_FRAMES     = 16
SPIN_FRAMES       = 240
ELEV0, AZIM0      = 15, 225
SEED              = 123

np.random.seed(SEED)

# ----------------- HELPERS -------------------
def circular_slice(a: np.ndarray, start: int, length: int):
    idx = (np.arange(start, start + length) % len(a)).astype(int)
    return a[idx], idx

def local_rank2_pod(segment: np.ndarray, use_centroid: bool = False):
    # affine reference
    u_ref = segment.mean(axis=0) if use_centroid else segment[0]
    shift = segment - u_ref
    _, _, vt = np.linalg.svd(shift, full_matrices=False)
    V = vt.T
    V_p = V[:, :2]                 # (3x2)
    V_s = V[:,  2:3]               # (3x1)
    q_p = shift @ V_p              # (m_i, 2)
    q_s = (shift @ V_s).ravel()    # (m_i,)
    return V_p, V_s, u_ref, q_p, q_s

def manifold_surface(u_ref, V_p, V_s, rbf_model, q_p_samples, pad=0.2, n=35):
    z1 = np.linspace(q_p_samples[:,0].min()-pad, q_p_samples[:,0].max()+pad, n)
    z2 = np.linspace(q_p_samples[:,1].min()-pad, q_p_samples[:,1].max()+pad, n)
    Z1, Z2 = np.meshgrid(z1, z2)
    q_s_grid = rbf_model(Z1, Z2)
    Surf_lin  = (V_p @ np.stack([Z1.ravel(), Z2.ravel()], axis=0)).reshape(3, *Z1.shape)
    Surf_full = u_ref[:, None, None] + Surf_lin + (V_s @ q_s_grid.ravel()[None, :]).reshape(3, *Z1.shape)
    return Surf_full  # (3, n, n)

# ------------- MAIN ANIMATION ----------------
def two_nonlinear_manifolds_gif(
    S: np.ndarray,
    overlap: float = 0.25,                # overlap fraction between the 2 windows
    kernel: str = 'multiquadric',         # or 'gaussian'
    smooth: float = 0.0,
    gif_name: str = GIF_NAME,
    fps: int = FPS
):
    m = len(S)

    # --- Build exactly TWO circular segments with overlap ---
    base_len = int(np.ceil((1 + overlap) * m / 2))
    half = base_len // 2
    centers = np.array([0, m//2], dtype=int)  # two opposite centers
    colors  = ['#9b59b6', '#27ae60']         # purple & green

    locals_data = []
    blend_sum = np.zeros_like(S, dtype=float)
    blend_cnt = np.zeros((m, 1), dtype=float)

    for seg_idx, c in enumerate(centers):
        seg, idxs = circular_slice(S, c - half, base_len)

        # Local POD split & reduced data
        V_p, V_s, u_ref, q_p, q_s = local_rank2_pod(seg, use_centroid=True)

        # Local closure 𝒩_{Φ,i} via RBF
        model = Rbf(q_p[:,0], q_p[:,1], q_s, function=kernel, smooth=smooth)

        # Local nonlinear reconstruction at segment samples
        q_s_hat = model(q_p[:,0], q_p[:,1])
        recon   = u_ref + q_p @ V_p.T + np.outer(q_s_hat, V_s.ravel())  # (len(seg), 3)

        # Local nonlinear surface
        Xp, Yp, Zp = manifold_surface(u_ref, V_p, V_s, model, q_p, pad=0.2, n=35)

        locals_data.append(dict(
            idxs=idxs, color=colors[seg_idx], recon=recon, Xp=Xp, Yp=Yp, Zp=Zp
        ))

        # Blend accumulators (uniform weights on overlaps)
        blend_sum[idxs] += recon
        blend_cnt[idxs] += 1.0

    S_blend = blend_sum / np.clip(blend_cnt, 1.0, None)

    # --------- Timeline ----------
    N_traj = m
    N_in   = SURF_FADE_FRAMES
    N_draw = max(LOCAL_DRAW_FRAMES, base_len)
    N_out  = LOCAL_FADE_FRAMES
    N_holdA = HOLD_A_FRAMES
    N_final_in = FINAL_SURF_FADE
    N_blend = max(BLEND_DRAW_FRAMES, m)
    N_holdB = HOLD_B_FRAMES
    N_spin  = SPIN_FRAMES

    phases = []
    t = 0
    phases.append(dict(kind="traj", start=t, end=t+N_traj)); t += N_traj
    for i in range(2):
        phases.append(dict(kind="manifold_in", seg=i, start=t, end=t+N_in)); t += N_in
        phases.append(dict(kind="local_draw",   seg=i, start=t, end=t+N_draw)); t += N_draw
        phases.append(dict(kind="manifold_out", seg=i, start=t, end=t+N_out)); t += N_out
    phases.append(dict(kind="holdA", start=t, end=t+N_holdA)); t += N_holdA
    # re-add both manifolds, kept
    for i in range(2):
        phases.append(dict(kind="final_in", seg=i, start=t, end=t+N_final_in)); t += N_final_in
    phases.append(dict(kind="blend_draw", start=t, end=t+N_blend)); t += N_blend
    phases.append(dict(kind="holdB", start=t, end=t+N_holdB)); t += N_holdB
    phases.append(dict(kind="spin", start=t, end=t+N_spin)); t += N_spin
    N_total = t

    # --------------- Figure & artists ----------------
    fig = plt.figure(figsize=(10, 8))
    ax  = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1.5, 1.5]); ax.set_ylim([-1.5, 1.5]); ax.set_zlim([-0.6, 0.6])
    ax.set_xlabel(r'$s_1$'); ax.set_ylabel(r'$s_2$'); ax.set_zlabel(r'$s_3$')
    ax.set_title(r'\textbf{Two local nonlinear manifolds} $(\mathcal{N}_{\Phi,1},\,\mathcal{N}_{\Phi,2})$', pad=18)
    ax.view_init(elev=ELEV0, azim=AZIM0)

    # True trajectory
    traj_line, = ax.plot([], [], [], 'o', color='black', markersize=4,
                         label=r'\textit{trajectory} $\mathbf{s}(t)$')

    # Local nonlinear reconstruction (overwritten color per segment)
    local_line, = ax.plot([], [], [], ls='--', lw=2, color='gray',
                          label=r'local nonlinear reconstruction')

    # Temporary surface (for i=0 then i=1)
    temp_surf = None

    # Final (persistent) surfaces
    final_surfs = [None, None]

    # Blended nonlinear curve
    blend_line, = ax.plot([], [], [], lw=3, color='crimson',
                          label=r'\textit{blended nonlinear approximation}')

    # Legend
    legend_elems = [
        Line2D([], [], marker='o', ls='None', color='k', label=r'\textit{trajectory} $\mathbf{s}(t)$'),
        Line2D([], [], ls='--', lw=2, color='#9b59b6', label=r'local recon on $\mathcal{N}_{\Phi,1}$'),
        Line2D([], [], ls='--', lw=2, color='#27ae60', label=r'local recon on $\mathcal{N}_{\Phi,2}$'),
        Line2D([], [], lw=3, color='crimson', label=r'\textit{blended nonlinear approximation}'),
        Patch(facecolor='#9b59b6', edgecolor='gray', alpha=0.30, label=r'nonlinear manifold $\mathcal{N}_{\Phi,1}$'),
        Patch(facecolor='#27ae60', edgecolor='gray', alpha=0.30, label=r'nonlinear manifold $\mathcal{N}_{\Phi,2}$'),
    ]
    ax.legend(handles=legend_elems, loc='upper right', frameon=True)

    # --------------- Update -------------------
    def init():
        nonlocal temp_surf
        traj_line.set_data([], []); traj_line.set_3d_properties([])
        local_line.set_data([], []); local_line.set_3d_properties([]); local_line.set_alpha(1.0)
        blend_line.set_data([], []); blend_line.set_3d_properties([])
        temp_surf = None
        ax.view_init(elev=ELEV0, azim=AZIM0)
        return traj_line, local_line, blend_line

    def find_phase(frame):
        for ph in phases:
            if ph["start"] <= frame < ph["end"]:
                return ph
        return phases[-1]

    def update(frame):
        nonlocal temp_surf
        ph = find_phase(frame)

        if ph["kind"] != "spin":
            ax.view_init(elev=ELEV0, azim=AZIM0)

        if ph["kind"] == "traj":
            k = frame - ph["start"] + 1
            traj_line.set_data(S[:k,0], S[:k,1])
            traj_line.set_3d_properties(S[:k,2])

        elif ph["kind"] == "manifold_in":
            seg = ph["seg"]
            rec = locals_data[seg]
            frac = (frame - ph["start"] + 1) / (ph["end"] - ph["start"])
            if frame == ph["start"]:
                # remove previous temp surface
                if temp_surf is not None:
                    try: temp_surf.remove()
                    except Exception: pass
                    temp_surf = None
                # draw new temp surface
                temp_surf = ax.plot_surface(rec["Xp"], rec["Yp"], rec["Zp"],
                                            color=rec["color"], edgecolor='gray',
                                            linewidth=0.3, alpha=0.0)
                local_line.set_color(rec["color"])
                local_line.set_data([], []); local_line.set_3d_properties([])
            if temp_surf is not None:
                temp_surf.set_alpha(SURF_ALPHA * frac)

        elif ph["kind"] == "local_draw":
            seg = ph["seg"]
            rec = locals_data[seg]
            k = frame - ph["start"] + 1
            k = int(min(k, len(rec["recon"])))
            local_line.set_data(rec["recon"][:k,0], rec["recon"][:k,1])
            local_line.set_3d_properties(rec["recon"][:k,2])
            if temp_surf is not None:
                temp_surf.set_alpha(SURF_ALPHA)

        elif ph["kind"] == "manifold_out":
            seg = ph["seg"]
            rec = locals_data[seg]
            prog = (frame - ph["start"] + 1) / (ph["end"] - ph["start"])
            frac = 1.0 - prog
            if temp_surf is not None:
                temp_surf.set_alpha(SURF_ALPHA * frac)
                if frame == ph["end"] - 1:
                    try: temp_surf.remove()
                    except Exception: pass
                    temp_surf = None
            if frame == ph["end"] - 1:
                local_line.set_data([], []); local_line.set_3d_properties([])

        elif ph["kind"] == "holdA":
            pass

        elif ph["kind"] == "final_in":
            seg = ph["seg"]
            rec = locals_data[seg]
            if final_surfs[seg] is None:
                final_surfs[seg] = ax.plot_surface(
                    rec["Xp"], rec["Yp"], rec["Zp"],
                    color=rec["color"], edgecolor='gray', linewidth=0.3, alpha=0.0
                )
            frac = (frame - ph["start"] + 1) / (ph["end"] - ph["start"])
            final_surfs[seg].set_alpha(SURF_ALPHA * frac)

        elif ph["kind"] == "blend_draw":
            k = frame - ph["start"] + 1
            K = min(k, len(S_blend))
            blend_line.set_data(S_blend[:K,0], S_blend[:K,1])
            blend_line.set_3d_properties(S_blend[:K,2])

        elif ph["kind"] == "holdB":
            pass

        elif ph["kind"] == "spin":
            frac = (frame - ph["start"]) / max(1, (ph["end"] - ph["start"] - 1))
            ax.view_init(elev=ELEV0, azim=AZIM0 + 360.0 * frac)

        return traj_line, local_line, blend_line

    ani = FuncAnimation(fig, update, frames=N_total, init_func=init, blit=False)
    ani.save(gif_name, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved GIF to: {gif_name}")

# ---------------- DEMO ----------------
if __name__ == "__main__":
    t = np.linspace(0, 2*np.pi, 100)
    S_demo = np.c_[np.cos(t), np.sin(t), 0.5*np.cos(2*t)].astype(float)

    two_nonlinear_manifolds_gif(
        S_demo,
        overlap=0.30,           # more visible overlap between the two locals
        kernel='multiquadric',  # try 'gaussian' as an alternative
        smooth=0.0,
        gif_name=GIF_NAME,
        fps=FPS
    )
