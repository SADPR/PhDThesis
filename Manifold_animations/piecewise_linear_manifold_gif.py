#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from itertools import cycle

# -------------------- STYLE --------------------
plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "axes.titlesize": 20,
    "axes.labelsize": 14,
    "xtick.labelsize": 14,
    "ytick.labelsize": 14,
    "legend.fontsize": 14
})

# ----------------- HELPERS ---------------------
def circular_slice(a: np.ndarray, start: int, length: int):
    idx = np.arange(start, start + length) % len(a)
    return a[idx], idx

def local_rank2_pod(segment: np.ndarray):
    s_ref = segment[0]
    shift = segment - s_ref
    _, _, vt = np.linalg.svd(shift, full_matrices=False)
    V = vt[:2].T                 # (3x2)
    z = shift @ V
    recon = s_ref + z @ V.T
    return recon, V, s_ref, z

# ------------- MAIN ANIMATION ------------------
def piecewise_manifold_gif(
    S: np.ndarray,
    n_seg: int = 4,
    overlap: float = 0.1,
    elev0: int = 15,
    azim0: int = 225,
    gif_name: str = "piecewise_linear_manifold.gif",
    fps: int = 24
):
    """
    Single GIF:
      • Animate the full trajectory first.
      • For each segment i=1..n_seg:
          - fade-in local plane i
          - draw local reconstruction i
          - fade-out plane i + clear local line
      • Final phase:
          - re-add planes 1..n_seg sequentially and keep them
          - draw blended approximated trajectory
          - final spin.
    """
    m = len(S)

    # Segment window length (with overlap) and centers on the circle
    seg_len = int(np.ceil((1 + overlap) * m / n_seg))
    half = seg_len // 2
    centres = np.linspace(0, m, n_seg, endpoint=False, dtype=int)

    # Colors for planes
    colours = ['skyblue', 'palegreen', 'plum', 'lightcoral', 'gold', 'lightsalmon', 'turquoise']
    col_cycle = cycle(colours)

    # Precompute local PODs and blending
    grid = np.linspace(-1.2, 1.2, 20)
    local = []
    blend_sum = np.zeros_like(S, dtype=float)
    blend_cnt = np.zeros((m, 1), dtype=float)

    for c in centres:
        seg, idxs = circular_slice(S, c - half, seg_len)
        recon, V, s_ref, z = local_rank2_pod(seg)

        # plane grid around segment z-mean
        z_mean = z.mean(axis=0)
        A, B = np.meshgrid(grid + z_mean[0], grid + z_mean[1])
        Zred = np.vstack((A.ravel(), B.ravel()))
        plane_pts = s_ref[:, None] + V @ Zred
        Xp = plane_pts[0].reshape(A.shape)
        Yp = plane_pts[1].reshape(A.shape)
        Zp = plane_pts[2].reshape(A.shape)

        color = next(col_cycle)
        local.append(dict(
            idxs=idxs, recon=recon, Xp=Xp, Yp=Yp, Zp=Zp, color=color
        ))

        blend_sum[idxs] += recon
        blend_cnt[idxs] += 1

    S_blend = blend_sum / np.clip(blend_cnt, 1, None)

    # --------- Timeline (phases & frames) ----------
    N_traj = m                               # animate trajectory points
    N_plane_fade = 15                        # fade-in frames
    N_local_draw_per_seg = max(30, seg_len)  # draw local recon (cap for smoothness)
    N_local_fadeout = 12                     # fade out plane + clear line
    N_hold_before_final = 12                 # pause
    N_final_plane_in = 18                    # fade-in per final plane (kept)
    N_blend_draw = m                         # draw blended recon
    N_hold2 = 18
    N_final_spin = 240                       # one full spin

    phases = []
    t = 0
    phases.append(dict(kind="traj", start=t, end=t+N_traj)); t += N_traj
    for i in range(n_seg):
        phases.append(dict(kind="plane_in", seg=i, start=t, end=t+N_plane_fade)); t += N_plane_fade
        phases.append(dict(kind="local_draw", seg=i, start=t, end=t+N_local_draw_per_seg)); t += N_local_draw_per_seg
        phases.append(dict(kind="local_out", seg=i, start=t, end=t+N_local_fadeout)); t += N_local_fadeout
    phases.append(dict(kind="holdA", start=t, end=t+N_hold_before_final)); t += N_hold_before_final
    for i in range(n_seg):
        phases.append(dict(kind="final_plane_in", seg=i, start=t, end=t+N_final_plane_in)); t += N_final_plane_in
    phases.append(dict(kind="blend_draw", start=t, end=t+N_blend_draw)); t += N_blend_draw
    phases.append(dict(kind="holdB", start=t, end=t+N_hold2)); t += N_hold2
    phases.append(dict(kind="final_spin", start=t, end=t+N_final_spin)); t += N_final_spin
    N_total = t

    # --------------- Figure & artists --------------------
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([-1.5, 1.5]); ax.set_ylim([-1.5, 1.5]); ax.set_zlim([-0.6, 0.6])
    ax.set_xlabel(r'$s_1$'); ax.set_ylabel(r'$s_2$'); ax.set_zlabel(r'$s_3$')
    ax.set_title(r'\textbf{Piecewise linear manifolds}', pad=20)
    ax.view_init(elev=elev0, azim=azim0)

    # Trajectory (black dots) — now animated (start empty)
    traj_line, = ax.plot([], [], [], 'o', color='black', markersize=4,
                         label=r'\textit{trajectory} $\mathbf{s}(t)$')

    # Local recon line (darkorange), reused per segment
    local_line, = ax.plot([], [], [], ls='--', lw=2, color='darkorange',
                          label=r'local reconstruction')
    local_line.set_alpha(1.0)

    # Dynamic per-segment plane (temporary; removed after segment)
    surf = None

    # Final planes (persistent)
    final_surfs = [None]*n_seg

    # Final blended recon (crimson)
    blend_line, = ax.plot([], [], [], lw=3, color='crimson',
                          label=r'\textit{approximated trajectory}')

    # -------- Legend with manifold entries (1..n_seg) --------
    legend_elems = [
        Line2D([], [], marker='o', ls='None', color='k', label=r'\textit{trajectory} $\mathbf{s}(t)$'),
        Line2D([], [], ls='--', lw=2, color='darkorange', label=r'local reconstruction'),
        Line2D([], [], lw=3, color='crimson', label=r'\textit{approximated trajectory}')
    ]
    for i, rec in enumerate(local, start=1):
        legend_elems.append(
            Patch(facecolor=rec['color'], edgecolor='gray', alpha=0.3,
                  label=rf'linear manifold {i}')
        )
    ax.legend(handles=legend_elems,
              loc='upper right',
              bbox_to_anchor=(1.2, 1.02),
              borderaxespad=0., frameon=True)

    # --------------- Update function ---------------------
    def init():
        traj_line.set_data([], []); traj_line.set_3d_properties([])
        local_line.set_data([], []); local_line.set_3d_properties([]); local_line.set_alpha(1.0)
        blend_line.set_data([], []); blend_line.set_3d_properties([])
        nonlocal surf; surf = None
        ax.view_init(elev=elev0, azim=azim0)
        return traj_line, local_line, blend_line

    def find_phase(frame):
        for ph in phases:
            if ph["start"] <= frame < ph["end"]:
                return ph
        return phases[-1]

    def update(frame):
        ph = find_phase(frame)

        # Default: static camera unless final spin
        if ph["kind"] != "final_spin":
            ax.view_init(elev=elev0, azim=azim0)

        if ph["kind"] == "traj":
            k = frame - ph["start"] + 1
            traj_line.set_data(S[:k, 0], S[:k, 1])
            traj_line.set_3d_properties(S[:k, 2])

        elif ph["kind"] == "plane_in":
            seg = ph["seg"]
            frac = (frame - ph["start"] + 1) / (ph["end"] - ph["start"])
            if frame == ph["start"]:
                nonlocal surf
                if surf is not None:
                    try: surf.remove()
                    except Exception: pass
                rec = local[seg]
                surf = ax.plot_surface(rec["Xp"], rec["Yp"], rec["Zp"],
                                       color=rec["color"], edgecolor='gray',
                                       linewidth=0.3, alpha=0.0)
                # Reset local line
                local_line.set_alpha(1.0)
                local_line.set_data([], []); local_line.set_3d_properties([])
            # Fade in plane
            surf.set_alpha(0.15 * frac)

        elif ph["kind"] == "local_draw":
            seg = ph["seg"]
            rec = local[seg]
            k = frame - ph["start"] + 1
            k = min(k, len(rec["recon"]))
            local_line.set_data(rec["recon"][:k, 0], rec["recon"][:k, 1])
            local_line.set_3d_properties(rec["recon"][:k, 2])
            if surf is not None:
                surf.set_alpha(0.15)

        elif ph["kind"] == "local_out":
            prog = (frame - ph["start"] + 1) / (ph["end"] - ph["start"])
            frac = 1.0 - prog
            if surf is not None:
                surf.set_alpha(0.15 * frac)
                if frame == ph["end"] - 1:
                    try: surf.remove()
                    except Exception: pass
                    surf = None
            if frame == ph["end"] - 1:
                local_line.set_data([], [])
                local_line.set_3d_properties([])
                local_line.set_alpha(1.0)

        elif ph["kind"] == "holdA":
            pass

        elif ph["kind"] == "final_plane_in":
            seg = ph["seg"]
            rec = local[seg]
            if final_surfs[seg] is None:
                final_surfs[seg] = ax.plot_surface(
                    rec["Xp"], rec["Yp"], rec["Zp"],
                    color=rec["color"], edgecolor='gray', linewidth=0.3, alpha=0.0
                )
            frac = (frame - ph["start"] + 1) / (ph["end"] - ph["start"])
            final_surfs[seg].set_alpha(0.15 * frac)

        elif ph["kind"] == "blend_draw":
            k = frame - ph["start"] + 1
            blend_line.set_data(S_blend[:k, 0], S_blend[:k, 1])
            blend_line.set_3d_properties(S_blend[:k, 2])

        elif ph["kind"] == "holdB":
            pass

        elif ph["kind"] == "final_spin":
            frac = (frame - ph["start"]) / max(1, (ph["end"] - ph["start"] - 1))
            ax.view_init(elev=elev0, azim=azim0 + 360.0 * frac)

        return traj_line, local_line, blend_line

    ani = FuncAnimation(fig, update, frames=N_total, init_func=init, blit=False)
    ani.save(gif_name, writer=PillowWriter(fps=fps))
    plt.close(fig)
    print(f"Saved GIF to: {gif_name}")

# ---------------- DEMO ----------------
if __name__ == "__main__":
    t_demo = np.linspace(0, 2*np.pi, 100)
    S_demo = np.c_[np.cos(t_demo), np.sin(t_demo), 0.5*np.cos(2*t_demo)]
    piecewise_manifold_gif(S_demo, n_seg=4, overlap=0.1)
