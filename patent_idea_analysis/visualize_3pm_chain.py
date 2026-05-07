"""
Step-by-step 3-PM signal chain visualization.

Loads a saved optimization result and produces 8 figures:
  1. CW input
  2. After PM1
  3. After dispersion 1
  4. Array — what each dispersed-PM1 combline contributes into PM2
  5. After PM2
  6. After dispersion 2
  7. Array — what each dispersed-PM2 combline contributes into PM3
  8. After PM3 (final output)

For the array figures each cell shows the comb that one input combline k
would generate out of the next PM: amplitude at harmonic n = B_k * J_{n-k}(beta).
The global phase of all sticks equals arg(B_k) and the m=0 stick (at n=k)
is highlighted.
"""

import json
import os

import matplotlib
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.special import jv

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
LOAD_PATH = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\projects\ao_patent_ideas\ssbm_by_cascaded_pm_and_dispersion\data\3stage_ssbm\20260425_153551"

DISPLAY_N_MAX = 3  # None = use n_max from config.json

USE_DB   = True
DB_FLOOR = -40.0

OTHER_ALPHA      = 0.40
HIGHLIGHT_N      = None    # harmonic to highlight in single-comb figures; None = no highlight
GLOBAL_PHASE_DEG = 180.0   # global phase rotation added to every combline in every plot [deg]

SHOW_ARRAY_FIGS = False   # set False to skip the per-combline contribution array figures

FOR_PUBLICATION = True   # suppress all titles and save each figure as an SVG
PUB_SAVE_PATH   = r"C:\Users\12242\OneDrive - UCB-O365\quantum_nanophoxonics\media"
PUB_GRID             = False   # also produce a composited 2×3 figure with all 6 stages
PUB_GRID_FIG_W_MM    = 210.0  # composite figure width  [mm]
PUB_GRID_FIG_H_MM    = 140.0  # composite figure height [mm]
PUB_GRID_DPI         = 200    # DPI used to rasterise each panel before compositing
PUB_GRID_H_GAP       = 0.01   # horizontal gap between panels (figure fraction)
PUB_GRID_V_GAP       = 0.01   # vertical   gap between panels (figure fraction)
# Per-panel view correction: shift azimuth/elevation by these amounts so panels
# look like a single camera view.  Left panels (col=0) get +AZIM_STEP; right
# panels (col=2) get -AZIM_STEP.  Top panels (row=0) get -ELEV_STEP/2; bottom
# panels (row=1) get +ELEV_STEP/2.  Set both to 0.0 for identical view angles.
PUB_GRID_AZIM_STEP   = 8.0   # azimuth degrees per column step
PUB_GRID_ELEV_STEP   = 4.0   # elevation degrees per row step

SHOW_FLOOR_SHADOW    = False   # project onto z = z_floor  (harmonic–Re plane)
SHOW_SIDE_SHADOW     = False   # project onto x = -N-0.5   (Re–Im phasor plane)
SHOW_BACK_SHADOW     = False  # project onto y = +ax_max  (harmonic–Im plane, far wall)
SHOW_DROP_LINES      = False  # draw dashed lines from ball to each shadow
SHADOW_ALPHA         = 0.15
SHADOW_BALL_SIZE     = 10
DROP_LINE_LW         = 0.6

SHOW_PHASOR_PANEL          = True    # 2-D Re-Im phasor diagram on a plane perpendicular to x-axis
PHASOR_PANEL_X             = None    # x position; None = right side (N + 0.5)
PHASOR_PANEL_CIRCLE        = True    # draw reference unit circle on the panel
PHASOR_PANEL_CIRCLE_LW     = 0.75
PHASOR_PANEL_CIRCLE_ALPHA  = 0.5
PHASOR_PANEL_STICK_LW      = 1.5
PHASOR_PANEL_BALL_SIZE     = 20
PHASOR_PANEL_ALPHA         = 0.4     # opacity for non-highlighted sticks; 1.0 when no highlight

SHOW_REFERENCE_RINGS    = True
RING_ALPHA              = 0.5
RING_LW                 = 0.75
RING_COLOR              = "#555555"   # ignored when RING_COLORED or RING_GRAYSCALE_GRADIENT
RING_COLORED            = False   # use each combline's own color instead of RING_COLOR
RING_GRAYSCALE_GRADIENT = True   # darkest at most-negative (front), lightest at most-positive (back)
RING_GRAD_DARK          = 0.45   # grayscale value for n = -N
RING_GRAD_LIGHT         = 0.85   # grayscale value for n = +N

SHOW_RING_POINTER    = True   # dashed line from origin to ring along each combline's phase
RING_POINTER_COLORED = True   # use each combline's own color for the pointer
RING_POINTER_LW      = 0.8
RING_POINTER_ALPHA   = 0.75
FREQ_AXIS_LW        = 0.75
FREQ_AXIS_TO_BACK   = True   # True = extend frequency axis all the way to x_back (back pane)

# Single-comb figure geometry (mm; converted to inches internally)
SINGLE_FIG_W_MM = 100.0
SINGLE_FIG_H_MM = 80.0
_MM_PER_INCH = 25.4
SINGLE_FIG_W = SINGLE_FIG_W_MM / _MM_PER_INCH
SINGLE_FIG_H = SINGLE_FIG_H_MM / _MM_PER_INCH
STICK_LW     = 1.5
BALL_SIZE    = 40
AXES_LINE_LW  = 1.2   # linewidth of the 3 main axis spine lines
HIDE_PANES    = False  # remove the grey axis pane backgrounds
BOX_EDGE_LW    = 0.5     # linewidth of the box edges (when HIDE_PANES=True)
BOX_EDGE_COLOR = "#000000"  # color of those edges
# Corner index whose 3 edges are omitted to give an open-box look.
# Corners are numbered 0-7:
#   0=(xl,yb,zb)  1=(xr,yb,zb)  2=(xr,yt,zb)  3=(xl,yt,zb)
#   4=(xl,yb,zt)  5=(xr,yb,zt)  6=(xr,yt,zt)  7=(xl,yt,zt)
# x_front = -(N+X_PAD_FRONT),  x_back = +(N+X_PAD_BACK),  yb/yt = ±ax_max*PAD,  zb/zt = ±ax_max*PAD
BOX_OPEN_CORNER = 4
X_PAD_FRONT  = 0.5   # gap from combline n=-N to front pane
X_PAD_BACK   = 3.5   # gap from combline n=+N to back pane (also default phasor panel position)
AXES_PAD     = 1.2   # axes y/z limits are this factor × ax_max so rings aren't clipped
ELEV         = 15
AZIM         = 180 + 40
ROLL         = 0
ORTHO_PROJ   = True   # True = orthographic projection; False = default perspective

# Array figure cell size and spacing (inches)
CELL_W_IN    = 2.2
CELL_H_IN    = 1.8
H_GAP_IN     = 0.05
V_GAP_IN     = 0.15
ARR_LEFT_IN  = 0.10
ARR_RIGHT_IN = 0.10
ARR_TOP_IN   = 0.25
ARR_BOT_IN   = 0.10

# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------
with open(os.path.join(LOAD_PATH, "config.json")) as f:
    cfg = json.load(f)
n_max_opt  = cfg["n_max"]
n_stages   = cfg["n_stages"]
poly_order = cfg.get("poly_order", None)
disp_n_max = cfg.get("disp_n_max", None)
assert n_stages == 3, f"Expected n_stages=3, got {n_stages}"

with open(os.path.join(LOAD_PATH, "result.json")) as f:
    res = json.load(f)
betas = res["betas"]  # [beta1, beta2, beta3]

phi_params_raw = np.load(os.path.join(LOAD_PATH, "phi_params.npy"), allow_pickle=True)

N           = DISPLAY_N_MAX if DISPLAY_N_MAX is not None else n_max_opt
harmonics   = np.arange(-N, N + 1)          # display window only
stage_range = np.arange(-n_max_opt, n_max_opt + 1)  # full computation range

# Build full-range phi arrays (length = len(stage_range))
def _build_phi_arr_full(raw):
    if poly_order is None:
        if disp_n_max is not None:
            phi_arr = np.zeros(len(stage_range))
            mask = (stage_range >= -disp_n_max) & (stage_range <= disp_n_max)
            phi_arr[mask] = raw
        else:
            phi_arr = np.asarray(raw, dtype=float)
    else:
        phi_arr = np.polyval(raw[::-1], stage_range)
        if disp_n_max is not None:
            phi_arr[np.abs(stage_range) > disp_n_max] = 0.0
    return phi_arr

phi1_full = _build_phi_arr_full(phi_params_raw[0])
phi2_full = _build_phi_arr_full(phi_params_raw[1])

# Slice a full-range array to the display window [-N, N]
def _to_display(full_arr):
    result = np.zeros(len(harmonics), dtype=complex)
    for i, n in enumerate(harmonics):
        if -n_max_opt <= n <= n_max_opt:
            result[i] = full_arr[int(n) + n_max_opt]
    return result

print(f"Loaded: betas={[f'{b:.4f}' for b in betas]}, n_max={n_max_opt}"
      + (f", poly_order={poly_order}" if poly_order is not None else "")
      + (f", disp_n_max={disp_n_max}" if disp_n_max is not None else ""))

# ---------------------------------------------------------------------------
# Drawing helpers
# ---------------------------------------------------------------------------
matplotlib.rcParams["font.family"] = "Arial"

_anchors = [
    (0.00, (200/255,  40/255,  60/255)),
    (0.12, (230/255,  80/255,  20/255)),
    (0.25, (240/255, 160/255,   0/255)),
    (0.40, (120/255, 200/255,  20/255)),
    (0.55, ( 20/255, 200/255, 140/255)),
    (0.70, ( 20/255, 120/255, 220/255)),
    (0.85, ( 80/255,  60/255, 220/255)),
    (1.00, (160/255,  20/255, 200/255)),
]
_cdict = {"red": [], "green": [], "blue": []}
for _pos, (_r, _g, _b) in _anchors:
    _cdict["red"].append((_pos, _r, _r))
    _cdict["green"].append((_pos, _g, _g))
    _cdict["blue"].append((_pos, _b, _b))
_cmap  = mcolors.LinearSegmentedColormap("spectral_comb", _cdict)
_cnorm = mcolors.Normalize(vmin=-N, vmax=N)


def _colors_for(harmonics):
    return [_cmap(_cnorm(n)) for n in harmonics]


def _blend_white(color, alpha):
    r, g, b = mcolors.to_rgb(color)
    return (1 - alpha + alpha * r, 1 - alpha + alpha * g, 1 - alpha + alpha * b)


def _display_amps(amplitudes):
    """Return (display_amplitudes, lengths) with dB-scaled or linear lengths."""
    mags = np.abs(amplitudes)
    if USE_DB:
        with np.errstate(divide="ignore"):
            pdb = 10.0 * np.log10(np.where(mags > 0, mags ** 2, 1e-30))
        lengths = np.clip((pdb - DB_FLOOR) / (-DB_FLOOR), 0, None)
    else:
        lengths = mags.copy()
    with np.errstate(invalid="ignore"):
        unit = np.where(mags > 0, amplitudes / mags, 0 + 0j)
    return lengths * unit, lengths


def _draw_on_ax(ax, amplitudes, highlight_n=None, ax_max_override=None,
                y_off=0.0, z_off=0.0, setup_ax=True,
                elev_override=None, azim_override=None):
    """Draw sticks+balls on a 3D axis. Returns ax_max.

    y_off / z_off shift every Re/Im coordinate so multiple stages can share
    one axes (combined grid figure).  setup_ax=False skips limits/view/panes.
    """
    disp_amps, lengths = _display_amps(amplitudes * np.exp(1j * np.deg2rad(GLOBAL_PHASE_DEG)))
    ax_max  = ax_max_override if ax_max_override is not None else (
        max(1.0, float(np.max(lengths))) if np.max(lengths) > 0 else 1.0
    )
    z_floor = -ax_max
    colors  = _colors_for(harmonics)

    x_front = -N - X_PAD_FRONT
    x_back  =  N + X_PAD_BACK

    # frequency-axis reference line
    x_axis_end = x_back if FREQ_AXIS_TO_BACK else N + X_PAD_FRONT
    ax.plot([x_front, x_axis_end], [y_off, y_off], [z_off, z_off],
            color="black", linewidth=FREQ_AXIS_LW, zorder=0)

    _theta = np.linspace(0, 2 * np.pi, 120)
    _rx, _ry = np.cos(_theta), np.sin(_theta)

    for n, A, c in zip(harmonics[::-1], disp_amps[::-1], colors[::-1]):
        z_base = (N - n) * 10
        if RING_GRAYSCALE_GRADIENT:
            g = RING_GRAD_DARK + (RING_GRAD_LIGHT - RING_GRAD_DARK) * (n + N) / (2 * N) if N > 0 else RING_GRAD_DARK
            rc = (g, g, g)
        elif RING_COLORED:
            rc = c
        else:
            rc = RING_COLOR
        re, im = float(A.real), float(A.imag)
        mag = abs(A)

        if SHOW_REFERENCE_RINGS:
            ax.plot([n] * len(_theta), _rx + y_off, _ry + z_off,
                    color=rc, linewidth=RING_LW, alpha=RING_ALPHA,
                    zorder=z_base + 1)

        if SHOW_RING_POINTER and mag > 0:
            ure, uim = re / mag, im / mag
            rpc = c if RING_POINTER_COLORED else rc
            ax.plot([n, n], [y_off, ure + y_off], [z_off, uim + z_off],
                    color=rpc, linewidth=RING_POINTER_LW,
                    linestyle="--", alpha=RING_POINTER_ALPHA,
                    zorder=z_base + 2)

        rgba_s = (*mcolors.to_rgb(c), SHADOW_ALPHA)
        x_side = x_front
        z_wall = z_floor * AXES_PAD + z_off
        y_wall = ax_max * AXES_PAD + y_off

        if SHOW_FLOOR_SHADOW:
            if SHOW_DROP_LINES:
                ax.plot([n, n], [re + y_off, re + y_off], [im + z_off, z_wall],
                        color="gray", linewidth=DROP_LINE_LW, linestyle="--",
                        zorder=z_base + 3)
            ax.scatter([n], [re + y_off], [z_wall],
                       color=[rgba_s], s=SHADOW_BALL_SIZE,
                       zorder=z_base + 3, depthshade=False)

        if SHOW_SIDE_SHADOW:
            if SHOW_DROP_LINES:
                ax.plot([n, x_side], [re + y_off, re + y_off], [im + z_off, im + z_off],
                        color="gray", linewidth=DROP_LINE_LW, linestyle="--",
                        zorder=z_base + 3)
            ax.scatter([x_side], [re + y_off], [im + z_off],
                       color=[rgba_s], s=SHADOW_BALL_SIZE,
                       zorder=z_base + 3, depthshade=False)

        if SHOW_BACK_SHADOW:
            if SHOW_DROP_LINES:
                ax.plot([n, n], [re + y_off, y_wall], [im + z_off, im + z_off],
                        color="gray", linewidth=DROP_LINE_LW, linestyle="--",
                        zorder=z_base + 3)
            ax.scatter([n], [y_wall], [im + z_off],
                       color=[rgba_s], s=SHADOW_BALL_SIZE,
                       zorder=z_base + 3, depthshade=False)

        if highlight_n is not None and n == highlight_n:
            ax.plot([n, n], [y_off, re + y_off], [z_off, im + z_off],
                    color=c, linewidth=STICK_LW, solid_capstyle="round",
                    alpha=1.0, zorder=z_base + 4)
            ax.scatter([n], [re + y_off], [im + z_off],
                       color=[c], s=BALL_SIZE, alpha=1.0,
                       zorder=z_base + 5, depthshade=False,
                       edgecolors="black", linewidths=1.0)
        else:
            alpha = OTHER_ALPHA if highlight_n is not None else 1.0
            ax.plot([n, n], [y_off, re + y_off], [z_off, im + z_off],
                    color=c, linewidth=STICK_LW, solid_capstyle="round",
                    alpha=alpha, zorder=z_base + 4)
            ax.scatter([n], [re + y_off], [im + z_off],
                       color=[_blend_white(c, alpha)], s=BALL_SIZE,
                       zorder=z_base + 5, depthshade=False, edgecolors="none")

    # 2-D phasor panel
    if SHOW_PHASOR_PANEL:
        x_panel = x_back if PHASOR_PANEL_X is None else PHASOR_PANEL_X
        if PHASOR_PANEL_CIRCLE:
            _theta_p = np.linspace(0, 2 * np.pi, 200)
            ax.plot([x_panel] * len(_theta_p),
                    np.cos(_theta_p) + y_off, np.sin(_theta_p) + z_off,
                    color=RING_COLOR, linewidth=PHASOR_PANEL_CIRCLE_LW,
                    alpha=PHASOR_PANEL_CIRCLE_ALPHA, linestyle="--", zorder=1)
        ax.scatter([x_panel], [y_off], [z_off], color="black", s=8,
                   zorder=1, depthshade=False)
        for n, A, c in zip(harmonics[::-1], disp_amps[::-1], colors[::-1]):
            re, im = float(A.real), float(A.imag)
            is_hl = (highlight_n is not None and n == highlight_n)
            ax.plot([x_panel, x_panel], [y_off, re + y_off], [z_off, im + z_off],
                    color=c, linewidth=PHASOR_PANEL_STICK_LW,
                    solid_capstyle="round", alpha=PHASOR_PANEL_ALPHA, zorder=2)
            ax.scatter([x_panel], [re + y_off], [im + z_off],
                       color=[_blend_white(c, PHASOR_PANEL_ALPHA)],
                       s=PHASOR_PANEL_BALL_SIZE, zorder=3, depthshade=False,
                       edgecolors="black" if is_hl else "none",
                       linewidths=0.8 if is_hl else 0)

    # Per-stage wireframe box (drawn whether or not setup_ax, so it appears in combined figures too)
    if HIDE_PANES:
        xl, xr = x_front, x_back
        yb, yt = -ax_max * AXES_PAD + y_off, ax_max * AXES_PAD + y_off
        zb, zt = z_floor * AXES_PAD + z_off, ax_max * AXES_PAD + z_off
        corners = [
            (xl, yb, zb), (xr, yb, zb), (xr, yt, zb), (xl, yt, zb),
            (xl, yb, zt), (xr, yb, zt), (xr, yt, zt), (xl, yt, zt),
        ]
        all_edges = [
            (0, 1), (1, 2), (2, 3), (3, 0),
            (4, 5), (5, 6), (6, 7), (7, 4),
            (0, 4), (1, 5), (2, 6), (3, 7),
        ]
        _corner_edges = {
            0: {(0,1),(3,0),(0,4)}, 1: {(0,1),(1,2),(1,5)},
            2: {(1,2),(2,3),(2,6)}, 3: {(2,3),(3,0),(3,7)},
            4: {(4,5),(7,4),(0,4)}, 5: {(4,5),(5,6),(1,5)},
            6: {(5,6),(6,7),(2,6)}, 7: {(6,7),(7,4),(3,7)},
        }
        skip = {frozenset(e) for e in _corner_edges.get(BOX_OPEN_CORNER, set())}
        edges = [e for e in all_edges if frozenset(e) not in skip]
        for i, j in edges:
            x0, y0, z0 = corners[i]
            x1, y1, z1 = corners[j]
            ax.plot([x0, x1], [y0, y1], [z0, z1],
                    color=BOX_EDGE_COLOR, linewidth=BOX_EDGE_LW, zorder=0)

    if not setup_ax:
        return ax_max

    ax.set_xlim(x_front, x_back)
    ax.set_ylim(-ax_max * AXES_PAD, ax_max * AXES_PAD)
    ax.set_zlim(z_floor * AXES_PAD, ax_max * AXES_PAD)
    ax.set_xlabel(""); ax.set_ylabel(""); ax.set_zlabel("")
    ax.set_xticks([]); ax.set_yticks([]); ax.set_zticks([])
    ax.tick_params(axis="both", which="both", length=0, labelsize=0)
    for axis in [ax.xaxis, ax.yaxis, ax.zaxis]:
        axis.line.set_linewidth(AXES_LINE_LW)
        if HIDE_PANES:
            axis.pane.fill = False
            axis.pane.set_edgecolor((0, 0, 0, 0))
            axis.pane.set_linewidth(0)
    if not HIDE_PANES:
        # The x-pane (perpendicular to frequency axis) is often transparent by
        # default; match it to the y/z pane gray so all three panes look uniform.
        ax.xaxis.pane.set_facecolor(ax.yaxis.pane.get_facecolor())

    ax.view_init(elev=elev_override if elev_override is not None else ELEV,
                 azim=azim_override if azim_override is not None else AZIM,
                 roll=ROLL)
    if ORTHO_PROJ:
        ax.set_proj_type("ortho")
    ax.set_box_aspect([2.5, 1, 1])

    return ax_max


def _single_fig(amplitudes, highlight_n=None, title=None, save_stem=None):
    fig = plt.figure(figsize=(SINGLE_FIG_W, SINGLE_FIG_H))
    ax  = fig.add_subplot(111, projection="3d")
    _draw_on_ax(ax, amplitudes, highlight_n=highlight_n)
    if title and not FOR_PUBLICATION:
        ax.set_title(title, fontsize=9, pad=2)
    fig.tight_layout()
    if FOR_PUBLICATION and save_stem:
        os.makedirs(PUB_SAVE_PATH, exist_ok=True)
        fig.savefig(os.path.join(PUB_SAVE_PATH, save_stem + ".svg"), format="svg")
    return fig


def _array_fig(amps_per_k, title=None, save_stem=None):
    """
    One 3-D cell per input combline k. Layout:
      top row: harmonics -N .. 0  (N+1 cells)
      bot row: harmonics  1 .. N  (N   cells, centred under top row)
    """
    top_ks = list(harmonics[:N + 1])   # -N, ..., 0
    bot_ks = list(harmonics[N + 1:])   #  1, ..., N

    n_top = len(top_ks)
    n_bot = len(bot_ks)

    top_w = n_top * CELL_W_IN + (n_top - 1) * H_GAP_IN
    bot_w = n_bot * CELL_W_IN + (n_bot - 1) * H_GAP_IN
    fig_w = ARR_LEFT_IN + top_w + ARR_RIGHT_IN
    fig_h = ARR_BOT_IN + CELL_H_IN + V_GAP_IN + CELL_H_IN + ARR_TOP_IN

    fig = plt.figure(figsize=(fig_w, fig_h))

    y0_top   = (ARR_BOT_IN + CELL_H_IN + V_GAP_IN) / fig_h
    y0_bot   = ARR_BOT_IN / fig_h
    bot_x_off = (top_w - bot_w) / 2.0   # centres bottom row under top row

    for ks, y0, x_off in [
        (top_ks, y0_top, 0.0),
        (bot_ks, y0_bot, bot_x_off),
    ]:
        for j, k in enumerate(ks):
            x0 = (ARR_LEFT_IN + x_off + j * (CELL_W_IN + H_GAP_IN)) / fig_w
            ax = fig.add_axes(
                [x0, y0, CELL_W_IN / fig_w, CELL_H_IN / fig_h],
                projection="3d",
            )
            _draw_on_ax(ax, amps_per_k[k], highlight_n=k)

    if title and not FOR_PUBLICATION:
        fig.suptitle(title, fontsize=10)
    if FOR_PUBLICATION and save_stem:
        os.makedirs(PUB_SAVE_PATH, exist_ok=True)
        fig.savefig(os.path.join(PUB_SAVE_PATH, save_stem + ".svg"), format="svg")

    return fig


def _grid_fig(stages):
    """2×3 grid of all 6 single-comb stages, sharing a common viewing angle.

    stages: list of 6 amplitude arrays in order
      [cw, after_pm1, after_disp1, after_pm2, after_disp2, after_pm3]
    """
    from io import BytesIO
    import matplotlib.image as mpimg

    # Shared scale across all panels
    global_max = 1.0
    for amps in stages:
        _, lengths = _display_amps(amps * np.exp(1j * np.deg2rad(GLOBAL_PHASE_DEG)))
        global_max = max(global_max, float(np.max(lengths)) if np.max(lengths) > 0 else 1.0)

    # panel layout: row 0 = top, row 1 = bottom; col 0-2 = left-right
    positions = [(col, row) for row in range(2) for col in range(3)]

    # Render each panel to an in-memory PNG with its corrected view angle
    panel_imgs = []
    for amps, (col, row) in zip(stages, positions):
        # Left panels (col=0) → higher azim; right panels (col=2) → lower azim.
        # Top panels (row=0) → lower elev; bottom panels (row=1) → higher elev.
        panel_azim = AZIM + (1 - col) * PUB_GRID_AZIM_STEP
        panel_elev = ELEV + (row - 0.5) * PUB_GRID_ELEV_STEP

        buf = BytesIO()
        fig_p = plt.figure(figsize=(SINGLE_FIG_W, SINGLE_FIG_H))
        ax_p  = fig_p.add_subplot(111, projection="3d")
        _draw_on_ax(ax_p, amps, highlight_n=HIGHLIGHT_N, ax_max_override=global_max,
                    elev_override=panel_elev, azim_override=panel_azim)
        fig_p.savefig(buf, format="png", dpi=PUB_GRID_DPI, bbox_inches="tight")
        plt.close(fig_p)
        buf.seek(0)
        panel_imgs.append(mpimg.imread(buf))

    # Composite the 6 rasters into a 2×3 2-D figure
    fig_w = PUB_GRID_FIG_W_MM / _MM_PER_INCH
    fig_h = PUB_GRID_FIG_H_MM / _MM_PER_INCH
    fig, axes = plt.subplots(2, 3, figsize=(fig_w, fig_h))
    for ax2d, img in zip(axes.flat, panel_imgs):
        ax2d.imshow(img)
        ax2d.axis("off")
    fig.subplots_adjust(wspace=PUB_GRID_H_GAP, hspace=PUB_GRID_V_GAP,
                        left=0, right=1, top=1, bottom=0)

    if FOR_PUBLICATION:
        os.makedirs(PUB_SAVE_PATH, exist_ok=True)
        fig.savefig(os.path.join(PUB_SAVE_PATH, "grid_all_stages.svg"), format="svg",
                    bbox_inches="tight")
    return fig


# ---------------------------------------------------------------------------
# Signal chain computation — all stages run on the full stage_range
# so that cross-stage Bessel mixing is exact; results are sliced to the
# display window [-N, N] only for plotting.
# ---------------------------------------------------------------------------

def _apply_pm_full(input_full, beta):
    """Full-range PM: A_n = sum_k B_k * J_{n-k}(beta) over all stage_range."""
    total = np.zeros(len(stage_range), dtype=complex)
    for k_idx, k in enumerate(stage_range):
        total += input_full[k_idx] * jv(stage_range - k, beta).astype(complex)
    return total


def _pm_contributions_display(input_disp, beta):
    """
    For each display combline k, compute the comb it generates through a PM.
    The n=k component is overridden with B_k (display only, array figures).
    Returns dict {k: complex array over harmonics}.
    """
    amps_per_k = {}
    for k_idx, k in enumerate(harmonics):
        B_k = input_disp[k_idx]
        amps = B_k * jv(harmonics - k, beta).astype(complex)
        k_pos = np.where(harmonics == k)[0]
        if len(k_pos):
            amps[k_pos[0]] = B_k
        amps_per_k[k] = amps
    return amps_per_k


# 1. CW input (full range)
cw_full = np.zeros(len(stage_range), dtype=complex)
cw_full[stage_range == 0] = 1.0

# 2. After PM1 (full range)
after_pm1_full = jv(stage_range, betas[0]).astype(complex)

# 3. After dispersion 1 (full range)
after_disp1_full = after_pm1_full * np.exp(1j * phi1_full)

# 5. After PM2 (full range)
after_pm2_full = _apply_pm_full(after_disp1_full, betas[1])

# 6. After dispersion 2 (full range)
after_disp2_full = after_pm2_full * np.exp(1j * phi2_full)

# 8. After PM3 (full range)
after_pm3_full = _apply_pm_full(after_disp2_full, betas[2])

# Slice all stages to display window
cw          = _to_display(cw_full)
after_pm1   = _to_display(after_pm1_full)
after_disp1 = _to_display(after_disp1_full)
after_pm2   = _to_display(after_pm2_full)
after_disp2 = _to_display(after_disp2_full)
after_pm3   = _to_display(after_pm3_full)

# Array-figure contributions (display range only, for visualisation)
pm2_contribs = _pm_contributions_display(after_disp1, betas[1])
pm3_contribs = _pm_contributions_display(after_disp2, betas[2])

# ---------------------------------------------------------------------------
# Generate all figures
# ---------------------------------------------------------------------------
_single_fig(cw,          highlight_n=HIGHLIGHT_N, title="1. CW input",           save_stem="01_cw_input")
_single_fig(after_pm1,   highlight_n=HIGHLIGHT_N, title=f"2. After PM1   β = {betas[0]:.3f} rad", save_stem="02_after_pm1")
_single_fig(after_disp1, highlight_n=HIGHLIGHT_N, title="3. After dispersion 1",  save_stem="03_after_disp1")
if SHOW_ARRAY_FIGS:
    _array_fig(pm2_contribs, title=f"4. PM2 (β = {betas[1]:.3f} rad) — contribution per input combline", save_stem="04_pm2_array")
_single_fig(after_pm2,   highlight_n=HIGHLIGHT_N, title=f"5. After PM2   β = {betas[1]:.3f} rad", save_stem="05_after_pm2")
_single_fig(after_disp2, highlight_n=HIGHLIGHT_N, title="6. After dispersion 2",  save_stem="06_after_disp2")
if SHOW_ARRAY_FIGS:
    _array_fig(pm3_contribs, title=f"7. PM3 (β = {betas[2]:.3f} rad) — contribution per input combline", save_stem="07_pm3_array")
_single_fig(after_pm3,   highlight_n=HIGHLIGHT_N, title=f"8. After PM3   β = {betas[2]:.3f} rad", save_stem="08_after_pm3")
if PUB_GRID:
    _grid_fig([cw, after_pm1, after_disp1, after_pm2, after_disp2, after_pm3])

plt.show()
